import os
import json
import logging
from typing import List, Tuple

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TherapySessionGenerator:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.api_key = api_key
        self.actions = ["hi", "angryWalking", "WalkingFast", "love", "lookingRight", "lookingLeft", "SlowlyHeadNod", "listening", "mad", "hug", "party", "oops", "sleep", "surprise", "suspicious", "terror", "think", "cute", "dizzy", "cryingFast", "cryingSlowly", "confused", "concerned", "admire", "angry", "paranoid"]
        self.face_exp = ["joy", "anger", "sadness", "amazement", "aggressiveness", "contempt", "love", "goofy", "sleepy", "surprise", "terror", "admiration", "fear", "disoriented", "starryEyes", "Ecstasy", "less rage", "more rage", "most range", "insane rage", "sleeping", "asleep"]
        self.arm_movement = ["left_arm_up", "left_arm_down", "left_arm_middle", "right_arm_up", "right_arm_down", "right_arm_middle", "both_arms_up", "both_arms_down", "both_arms_middle"]
        self.head_movement = ["head_middle", "head_slightly_left", "head_slightly_right", "head_full_left", "head_full_right", "head_tilt_left", "head_tilt_right"]

        self._initialize_models()
        self._initialize_prompts()

    def _initialize_models(self):
        CD_MODEL = 'gpt-4o-mini'
        POLICY_MODEL = 'gpt-4o-mini'
        THERAPY_MODEL = 'gpt-4o-mini'

        self.agent_cd_llm = ChatOpenAI(model=CD_MODEL, temperature=0.7)
        self.agent_policy_llm = ChatOpenAI(model=POLICY_MODEL, temperature=0.7)
        self.agent_therapy_llm = ChatOpenAI(model=THERAPY_MODEL, temperature=0.7)

    def _initialize_prompts(self):
        self.cd_prompt_template = PromptTemplate(
            input_variables=["therapy"],
            template="""
            **Task:** Create a comprehensive description of a therapy scene tailored to a specific speech and language disorder. The description should include the following elements:
        
            1. **Scene Context:**
               - Type of therapy {therapy}.
               - Purpose of the session.
        
            2. **Environment:**
               - Description of the therapy room (size, lighting, color scheme, furniture).
               - Any specific design elements that facilitate therapy.
        
            3. **Student Details:**
               - **Name:** [Choose a realistic name].
               - **Age:** [Specify age appropriate to the therapy].
               - **Gender:** [Specify gender].
               - **Speech Language Disorder:** [e.g., Expressive Language Disorder, Receptive Language Disorder].
        
            4. **Visual Scenes:**
               - **Student's View:** Describe what the student sees in the room (e.g., posters on the wall, toys on the shelf).
               - **Therapist's View:** Describe what the therapist sees (e.g., a laptop on the desk, a set of flashcards).
        
            5. **Props Needed:**
               - List and describe any props required for the therapy session (e.g., laptop, toys, picture cards, storybooks).
        
            **Constraints:**
            - Ensure the description is vivid and detailed to provide a clear mental image.
            - The environment should be conducive to the specific type of therapy.
            - Props should be relevant and support the therapy objectives.
        
            **Format:**
            Provide the description in a structured format with clear headings for each section.
            """
        )

        self.policy_prompt_template = PromptTemplate(
            input_variables=["therapy_scene_description"],
            template="""
            **Task:** Develop a detailed therapy session plan based on the provided therapy scene description. The plan should include objectives, materials needed, a step-by-step breakdown of activities with allocated time, strategies for handling potential edge cases, and steps to conclude the session effectively.
        
            **Input:**
            {therapy_scene_description}
        
            **Requirements:**
        
            1. **Session Overview:**
               - Brief summary of the session's purpose and focus.
        
            2. **Objectives:**
               - List clear, measurable goals for the session.
        
            3. **Materials Needed:**
               - Enumerate all props and materials required, referencing those mentioned in the scene description.
        
            4. **Step-by-Step Activities:**
               - **Activity 1:** [Name]
                 - **Duration:** [e.g., 10 minutes]
                 - **Description:** Detailed instructions on conducting the activity.
                 - **Purpose:** Explain how it meets the session objectives.
               - **Activity 2:** [Name]
                 - **Duration:** [e.g., 15 minutes]
                 - **Description:** ...
                 - **Purpose:** ...
               - *(Continue as needed for all activities in the session)*
        
            5. **Edge Case Handling:**
               - Identify potential challenges or deviations that might occur during each activity.
               - Provide strategies to address these scenarios effectively.
        
            6. **Timing and Flow:**
               - Allocate time for each activity to ensure the session remains on schedule.
               - Ensure a logical progression from one activity to the next.
        
            7. **Feedback and Reflection:**
               - Incorporate moments for providing feedback to the student.
               - Allow time for the student to reflect on their progress.
        
            8. **Homework Assignment:**
               - Suggest a task or activity for the student to practice skills learned during the session.
        
            9. **Conclusion:**
               - Summarize the session’s achievements.
               - Provide a positive and motivating end to the session.
        
            **Constraints:**
            - The total session duration should be realistic (e.g., 60 minutes) unless specified otherwise.
            - Activities should be engaging and appropriate for the student’s age and disorder.
            - Ensure flexibility to accommodate the student’s responsiveness and energy levels.
        
            **Format:**
            Present the session plan in a structured format with clear headings and subheadings for each section.
            """
        )

        self.therapy_prompt_template = PromptTemplate(
            input_variables=["session_plan"],
            template="""
            **Task:** Convert the provided therapy session plan into a detailed script of conversations between the therapist and the student. Include visual feedback about the environment as background info. Ensure that the dialogue adheres strictly to the session plan, reflects the planned activities, and maintains the session's objectives. The conversations should be natural, engaging, and appropriate for the student's age and disorder.
        
            **Input:**
            {session_plan}
        
            **Requirements:**
        
            1. **JSON Format:**
            Each conversation entry must be structured in JSON format as follows:
            {{
                "Therapist": "Therapist's dialogue",
                "Student": "Student's dialogue",
                "Background": "Background or scene description (if any, otherwise leave empty)"
            }} 
        
            2. **Structured Dialogue:**
               - Divide the script according to the session's step-by-step activities.
               - Clearly indicate who is speaking (Therapist or Student).
        
            3. **Natural Conversations:**
               - Ensure the dialogue flows naturally, reflecting realistic interactions.
               - Incorporate pauses, questions, and responses that a therapist and student might naturally use.
        
            4. **Adherence to Objectives:**
               - Align conversations with the session’s objectives, ensuring each interaction serves a purpose.
        
            5. **Handling Edge Cases:**
               - Preemptively include responses and strategies for potential edge cases as outlined in the session plan.
               - Demonstrate how the therapist manages unexpected student behaviors or deviations.
        
            6. **Consistency with Session Flow:**
               - Follow the chronological order of activities.
               - Allocate appropriate dialogue time for each activity based on the session's timing.
        
            7. **Engagement and Motivation:**
               - Include positive reinforcement, encouragement, and constructive feedback.
               - Maintain a supportive and motivating tone throughout the session.
        
            **Example Output:**
            [
                {{
                    "Therapist": "Good morning, Jamie! How are you feeling today?",
                    "Student": "I’m feeling good!",
                    "Background": "Jamie sits comfortably in his chair, smiling."
                }},
                {{
                    "Therapist": "Great! Let’s start by identifying emotions. What does this face show?",
                    "Student": "It looks happy!",
                    "Background": ""
                }},
                {{
                    "Therapist": "Exactly! Now, can you find a situation where someone might feel happy?",
                    "Student": "Sharing toys with friends.",
                    "Background": "Jamie looks at the emotion card and points to the matching picture card."
                }}
            ]
        
            **Constraints:**
            - Only include the dialogue and background information in JSON format.
            - Do not include activity headings or any additional narrative text.
            """
        )

    def _create_pipelines(self):
        self.cd_pipeline = self.cd_prompt_template | self.agent_cd_llm
        self.policy_pipeline = self.policy_prompt_template | self.agent_policy_llm
        self.therapy_pipeline = self.therapy_prompt_template | self.agent_therapy_llm

    def generate_scene_and_plan(self, therapy: str) -> Tuple[str, str, str]:
        logger.info(f"Generating data for the therapy: {therapy}")

        try:
            logger.info("Invoking Context Generation")
            therapy_scene_description = self.cd_pipeline.invoke({"therapy": therapy,"actions":self.actions,"expressions":self.face_exp,"move_arms":self.arm_movement,"move_head":self.head_movement}).content
            logger.info("Invoking Policy Agent")
            session_plan = self.policy_pipeline.invoke({"therapy_scene_description": therapy_scene_description}).content
            logger.info("Invoking Therapy Agent")
            conversational_script = self.therapy_pipeline.invoke({"session_plan": session_plan}).content

            return therapy_scene_description, session_plan, conversational_script

        except Exception as e:
            logger.error(f"Error generating therapy session data: {e}")
            raise

    def save_outputs(self, data: List[dict], json_path: str, txt_path: str):
        try:
            logger.info(f"Saving data to {json_path} and {txt_path}")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            logger.info(f"JSON data successfully saved to {json_path}")

            with open(txt_path, "w", encoding="utf-8") as txt_file:
                for entry in data:
                    formatted_output = f"""
### Therapy Scene Description ###

{entry['therapy_scene_description']}

### Session Plan ###

{entry['session_plan']}

## Conversation ##

{entry['convo_script']}
"""
                    txt_file.write(formatted_output)
            logger.info(f"Formatted text data successfully saved to {txt_path}")

        except IOError as e:
            logger.error(f"File writing failed: {e}")
            raise


def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    generator = TherapySessionGenerator(api_key=OPENAI_API_KEY)
    generator._create_pipelines()
    """
    therapies = [
        "Pragmatic Language Therapy(Talking in a formal setting)",
        "Pragmatic Language Therapy(Exchanging greetings with proper etiquettes)",
        "Receptive Language Therapy (Hanen Program)",
        "Receptive Language Therapy (Story Retelling)",
        "Semantic Intervention (sorting animals by habitat)",
        "Semantic Intervention (sorting objects by purpose)"
    ]
    """
    therapies = ["Pragmatic Language Therapy(Talking in a formal setting)"]
    output_file = "therapy_sess_example2.json"
    txt_output_file = "therapy_sess_example2.txt"

    total_gen = []
    for therapy in therapies:
        try:
            scene_desc, plan, dialog = generator.generate_scene_and_plan(therapy)
            output_data = {
                "Therapy": therapy,
                "therapy_scene_description": scene_desc,
                "session_plan": plan,
                "convo_script": dialog
            }
            total_gen.append(output_data)
            logger.info(f"Generated data for therapy: {therapy}")

        except Exception as e:
            logger.error(f"Failed to generate data for therapy '{therapy}': {e}")

    generator.save_outputs(total_gen, output_file, txt_output_file)


if __name__ == "__main__":
    main()
