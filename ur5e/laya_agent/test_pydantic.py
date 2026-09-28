import os
os.environ["HF_HUB_OFFLINE"] = "1"
from pydantic import BaseModel, Field
from laya import Router
from laya.structured import decide

class Action(BaseModel):
    dx_cm: int = Field(description="Delta X in cm (-3 to 3). Negative is left, positive is right.", ge=-3, le=3)
    dy_cm: int = Field(description="Delta Y in cm (-3 to 3). Negative is back, positive is forward.", ge=-3, le=3)
    dz_cm: int = Field(description="Delta Z in cm (-3 to 3). Negative is down, positive is up.", ge=-3, le=3)
    close_gripper: bool = Field(description="Should the gripper be closed?")

router = Router()
state = '{"dist_xy_meters": 0.0, "height_above_block_meters": 0.02, "gripper_open": true}'

res = decide(router, state, schema=Action, model="multilingual")
print(res)
