import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from planners.base_planner import BasePlanner
from utils.llm_util import SLLMBase
from utils.io_util import load_txt
from utils.tamp_util import text_to_actions, PrimitiveAction, Action, TAMPFeedback

logger = logging.getLogger(__name__)


class SLLMPlanner(BasePlanner, SLLMBase):
    """SLLM TAMP Planner"""

    def __init__(
        self,
        planner_prompt_file: str,        
        primitive_actions: Dict[str, PrimitiveAction],
        with_mp_feedback: bool = True,        
        *args,
        **kwargs,
    ):
        BasePlanner.__init__(self, primitive_actions=primitive_actions, *args, **kwargs)
        SLLMBase.__init__(self, use_gpt_4=True, *args, **kwargs)

        # load planning prompt template
        prompt_template_folder = Path(__file__).resolve().parent.parent / "prompts"
        planning_prompt_template = load_txt(
            prompt_template_folder / "planners" / f"{planner_prompt_file}.txt"
        )
                
        self._planning_prompt = planning_prompt_template

        self.with_mp_feedback = with_mp_feedback        

        # trace
        self._trace = []

    def reset(self):
        self._trace = []    

    def plan(
        self,
        obs,
        feedback_list: List[Tuple[Action, TAMPFeedback]],
        symbolic_plan: List[str] = [],
        *args,
        **kwargs,
    ):
        # prepare feedback
        if self.with_mp_feedback:
            feedback_text = ", ".join(
                [
                    f"{str(action)}(Motion: {feedback.motion_planner_feedback})"
                    for action, feedback in feedback_list
                ]
            )
        else:
            feedback_text_list = []
            for action, feedback in feedback_list:
                if feedback.action_success:
                    feedback_text_list.append(f"Motion: {str(action)}(Success)")
                else:
                    feedback_text_list.append(f"Motion: {str(action)}(Failure)")

            feedback_text = ", ".join(feedback_text_list)

        if len(feedback_list) > 0 and feedback_list[-1][1].action_success:
            feedback_text += f"(Task: {feedback_list[-1][1].task_process_feedback})"

        # plan        
        planning_prompt = self._planning_prompt
        planning_prompt = planning_prompt.replace("{red_pos}", str(obs['red_box']['position']).replace('(','[').replace(')',']'))
        planning_prompt = planning_prompt.replace("{blue_pos}", str(obs['blue_box']['position']).replace('(','[').replace(')',']'))
        planning_prompt = planning_prompt.replace("{green_pos}", str(obs['green_box']['position']).replace('(','[').replace(')',']'))
        planning_prompt = planning_prompt.replace("{tan_pos}", str(obs['tan_box']['position']).replace('(','[').replace(')',']'))

        # question = "Turn on the TV."        
        question = input("###########################################\n###########################################\nQuestion: ")
        planning_prompt = planning_prompt.replace("{question}", question)        

        plan_iter = 0
        plan, reasoning = None, None

        llm_output = self.prompt_llm(planning_prompt)        
        llm_output = llm_output.replace('remote','red_box')
        llm_output = llm_output.replace('spoon','blue_box')
        llm_output = llm_output.replace('cup','green_box')
        llm_output = llm_output.replace('phone','tan_box')        
        llm_output = json.loads(llm_output)        

        plan = text_to_actions(llm_output["Full Plan"], self._primitive_actions)
        reasoning = llm_output["Reasoning"]

        # save last feedback & reasoning to trace
        if plan is None:
            return None
        else:
            self._trace.append((feedback_text, reasoning))

        # import pdb

        # pdb.set_trace()

        return plan
