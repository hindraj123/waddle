
import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path(
    "/home/sra/waddle/humanoid/Humanoid.xml
)

data = mujoco.MjData(model)

viewer.launch(model)
