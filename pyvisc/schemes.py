import enum
from typing import Literal, List, Optional
import os

import jinja2
from pydantic import BaseModel, validator, Field


_path_templates = "./pyvisc/templates"
_path_save = "./pyvisc/generated"
env = jinja2.Environment(loader=jinja2.FileSystemLoader(_path_templates), extensions=["jinja2.ext.do"])


class _BaseScheme(BaseModel):
    """ Base class for all schemes """

    _filename: str
    _template_filename: str

    @property
    def filename(self):
        return self._filename

    @property
    def template_filename(self):
        return self._template_filename

    # Validator to transform every enum into str
    @validator("*", always=True)
    def transform_enum_to_value(cls, v):
        if isinstance(v, (enum.Enum, enum.IntEnum)):
            return v.value
        return v
    
    def render(self, path_save: str = _path_save) -> str:
        global env
        template = env.get_template(self.template_filename)
        rendered = template.render(**self.dict())

        filename_save = os.path.join(path_save, self.filename)
        if(not os.path.exists(os.path.dirname(filename_save))):
            os.makedirs(os.path.dirname(filename_save))
        with open(filename_save, "w") as f:
            f.write(rendered)

        return rendered


class SchemeVarH(_BaseScheme):
    _filename = "var.h"
    _template_filename = "var.h.j2"

    precision: Literal["single", "double"]
    NX: int
    NY: int
    NZ: int
    tau: float
    vel_set: Literal["D3Q19", "D3Q27"]
    use_ibm: bool
    non_newtonian_model: Optional[str]
    sim_id: int
    path_save: str
    steps: int
    macr_save: int
    data_report: int
    FX: float = 0
    FY: float = 0
    FZ: float = 0
    data_stop: bool = False
    data_save: bool = False
    pop_save: bool = False
    ini_step: int = 0
    n_gpus: int = 1
    resid_max: float = 1e-5
    


class SchemeIBMVarH(_BaseScheme):
    _filename = "IBM/ibmVar.h"
    _template_filename = "ibmVar.h.j2"
