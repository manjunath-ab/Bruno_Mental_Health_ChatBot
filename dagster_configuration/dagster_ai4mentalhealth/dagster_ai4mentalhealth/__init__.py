from dagster import Definitions, load_assets_from_modules, ScheduleDefinition, AssetSelection, define_asset_job
from . import blurt,nat,chipur


all_assets = load_assets_from_modules([blurt,nat,chipur])
#blurt_assets = load_assets_from_modules([blurt])
blurt_pipeline = define_asset_job("blurt_pipeline", selection=AssetSelection.groups("blurt_assets"))
blurt_schedule = ScheduleDefinition(job=blurt_pipeline, cron_schedule="11 2 * * *")

#nat_assets = load_assets_from_modules([nat])
nat_pipeline = define_asset_job("nat_pipeline", selection=AssetSelection.groups("nat_assets"))
nat_schedule = ScheduleDefinition(job=nat_pipeline, cron_schedule="11 2 * * *")
all_jobs = [blurt_pipeline,nat_pipeline]

chipur_pipeline = define_asset_job("chipur_pipeline", selection=AssetSelection.groups("chipur_assets"))
chipur_schedule = ScheduleDefinition(job=chipur_pipeline, cron_schedule="11 2 * * *")
all_jobs = [blurt_pipeline,nat_pipeline,chipur_pipeline]


defs = Definitions(
    assets=all_assets,
    jobs=all_jobs
    #schedules=[nat_schedule]
)
