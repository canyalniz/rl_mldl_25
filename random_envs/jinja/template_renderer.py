"""
This file was taken from the official random_envs repository https://github.com/gabrieletiboni/random-envs
This file is included in the main project repository to be able to submit a self contained code base.

@misc{tiboniadrbenchmark,
    title={Online vs. Offline Adaptive Domain Randomization Benchmark},
    author={Tiboni, Gabriele and Arndt, Karol and Averta, Giuseppe and Kyrki, Ville and Tommasi, Tatiana},
    year={2022},
    primaryClass={cs.RO},
    publisher={arXiv},
    doi={10.48550/ARXIV.2206.14661}
}
"""

import jinja2
import os
from math import sin, cos
import numpy as np

class TemplateRenderer():
    def __init__(self):
        # Set the relative template dir as base
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, "assets")
        self.template_dirs = [template_dir]

        self.loader = jinja2.FileSystemLoader(searchpath=self.template_dirs)
        self.template_env = jinja2.Environment(loader=self.loader)

    def render_template(self, template_file, **kwargs):
        template = self.template_env.get_template(template_file)
        rendered_xml = template.render(**kwargs, sin=sin, cos=cos, pi=np.pi)
        return rendered_xml

    def render_to_file(self, template_file, target_file, **kwargs):
        xml = self.render_template(template_file, **kwargs)
        with open(target_file, "w") as f:
            f.write(xml)

