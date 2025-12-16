import json
from typing import Iterable

import lazynwb
import polars as pl
import polars_ds as pds
import polars_vec_ops as vec
import numpy as np
import pydantic_settings
import pydantic
import tqdm
import upath

import utils

PSTH_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/psths')
NEURAL_TRAJ_DIR = upath.UPath('s3://aind-scratch-data/dynamic-routing/neural_trajectory_separation_hit_fa')

class Params(pydantic_settings.BaseSettings):
    name: str | None = pydantic.Field(None, exclude=True)
    skip_existing: bool = pydantic.Field(True, exclude=True)
    n_resample_iterations: int = 100

    # set the priority of the input sources:
    @classmethod  
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # instantiating the class will use arguments passed directly, or provided via the command line/app panel
        # the order of the sources below defines the priority (highest to lowest):
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )

units = utils.get_df('units')


def sessionwise_trajectory_distances(lf: pl.LazyFrame, label_1: str, label_2: str, group_by: str | Iterable[str] | None = None, streaming: bool = True) -> pl.DataFrame:
    if isinstance(lf, pl.DataFrame):
        streaming = False
    lf = lf.lazy()
    if group_by is None:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]
    group_by = tuple(group_by)
    return (
        lf
        .filter(pl.col('context_state').is_in([label_1, label_2]))
        .group_by('unit_id', 'context_state', *group_by)
        .agg(pl.col('psth').first()) # should only be one psth)
        .collect(engine='streaming' if streaming else 'auto')
        .pivot(on='context_state', values='psth')
        .with_columns(
            pl.col(label_1).sub(label_2).list.eval(pl.element().pow(2)).alias('diff^2')
        )
        .group_by(*group_by or ['unit_id'])
        .agg(
            pl.all(),
            pl.lit(f"{label_1}_vs_{label_2}").alias('description'),
            vec.sum('diff^2').list.eval(pl.element().sqrt()).truediv(pl.col('unit_id').count().sqrt()).cast(pl.List(pl.Float64)).alias('traj_separation'),
            # ^ cast ensures compat with any list[null] 
        )
        .drop('diff^2', label_1, label_2)
    )

def write_neural_trajectories(psth_dir: upath.UPath, params: Params) -> None:
    root_dir = NEURAL_TRAJ_DIR / psth_dir.name

    # write full set of trajectory separation data for each area
    for psth_path in psth_dir.glob('*.parquet'):
        area = psth_path.stem
        all_traj_sep_path = root_dir / f"{area}.parquet"
        if params.skip_existing and all_traj_sep_path.exists():
            print(f'Skipping {area}: parquet already on S3')
            continue

        lf = pl.scan_parquet(psth_path.as_posix())

        def resample_units(lf: pl.LazyFrame, seed: int) -> pl.LazyFrame:
            return (
                lf
                .sort('unit_id') # sorting and maintaining order critical to ensure same unit sample for each context
                .group_by('session_id', 'context_state', maintain_order=True)
                .agg(pl.all().sample(fraction=1, with_replacement=True, seed=seed))
                .explode(pl.all().exclude('session_id', 'context_state'))
            )
        null_iter = pl.col('null_iteration').is_null()
        named_lfs = {
            'actual': lf.filter(null_iter),
            'null': lf.filter(~null_iter),
            'resampled units': lf.filter(null_iter),
        }

        vis_hit_expr = pl.col('is_vis_target') & pl.col('is_hit')
        aud_hit_expr = pl.col('is_aud_target') & pl.col('is_hit')
        vis_confident_false_alarm_expr =  pl.col('is_vis_target') & pl.col('is_false_alarm') & pl.col('predict_proba').is_in(["(-inf, 0.2]", "(0.2, 0.4]"])
        aud_confident_false_alarm_expr =  pl.col('is_aud_target') & pl.col('is_false_alarm') & pl.col('predict_proba').is_in(["(0.6, 0.8]", "(0.8, inf]"])

        name_df_context_pair: list[tuple[str, pl.DataFrame, tuple[str, str]]] = []
        for name, named_lf in named_lfs.items():
            for label_1, context_1, label_2, context_2 in [
                ('vis_hit', vis_hit_expr, 'vis_fa', vis_confident_false_alarm_expr), 
                ('aud_hit', aud_hit_expr, 'aud_fa', aud_confident_false_alarm_expr),
            ]:
                print(f"Processing: {area} | {name} trajectories | {label_1} vs {label_2}")
                n = params.n_resample_iterations if name == 'resampled units' else 1
                if name == 'resampled units':
                    # fetch df to avoid reading 100 times
                    named_lf = named_lf.collect().lazy()
                named_lf = named_lf.with_columns(pl.when(context_1).then(pl.lit(label_1)).when(context_2).then(pl.lit(label_2)).alias('context_state'))
                for i in range(n):
                    if name == 'resampled units':
                        named_lf = named_lf.pipe(resample_units, seed=i)
                    df = sessionwise_trajectory_distances(named_lf, label_1=label_1, label_2=label_2, group_by=['session_id', 'null_iteration'], streaming=True)
                    name_df_context_pair.append((name, df, (label_1, label_2)))

#! TODO FIX FILTERING FOR NULL

        # calculate average null for each session:
        null_avgs = (
            pl.concat([df for name, df, _ in name_df_context_pair if name == 'null'])
            .group_by('session_id', 'description')
            .agg(
                vec.avg('traj_separation').alias('avg_null_traj_separation'),
            )
        )

        # store other dfs, with an additional null subtracted column
        dfs: list[pl.DataFrame] = []
        for name, df, _ in name_df_context_pair:
            if name == 'null':
                continue
            dfs.append(
                df
                .drop('null_iteration')
                .join(null_avgs, on=['session_id', 'description'], how='inner')
                .with_columns(
                    null_subtracted_traj_separation=pl.col('traj_separation') - pl.col('avg_null_traj_separation'),
                )
            )

        print(f"Writing {all_traj_sep_path.as_posix()}")
        (
            pl.concat(dfs)
            .with_columns(pl.lit(area).alias('area'))
        ).write_parquet(all_traj_sep_path.as_posix())


if __name__ == "__main__":

    params = Params()
    if params.name:
        psth_dirs = [PSTH_DIR / params.name]
        if not psth_dirs[0].exists():
            raise FileNotFoundError(f"PSTH directory does not exist: {psth_dirs[0]}")
    else:
        psth_dirs = list(d for d in PSTH_DIR.glob('*') if d.is_dir() if d.with_suffix('.json').exists())
        if not psth_dirs:
            raise FileNotFoundError(f"No valid PSTH directories found in {PSTH_DIR}")

    for i, psth_dir in enumerate(sorted(psth_dirs, reverse=True)):
        print(f"{i+1}/{len(psth_dirs)} | Processing PSTHs in {psth_dir.name}")

        original_json = json.loads((PSTH_DIR / f'{psth_dir.name}.json').read_text())
        new_json_path = NEURAL_TRAJ_DIR / f'{psth_dir.name}.json'
        # new_json_path.write_text(json.dumps(original_json | params.model_dump(), indent=4))

        write_neural_trajectories(psth_dir, params)

    print(f"All finished")