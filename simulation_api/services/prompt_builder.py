"""
Prompt builder for patient simulation.

Constructs LLM prompts from patient cases for realistic patient roleplay.
"""

from pathlib import Path

from simulation_api.schemas import PatientCase


def load_base_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "resources" / "simulation.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_patient_prompt(case: PatientCase) -> str:
    base = load_base_prompt()
    base = base.replace("{patient_name}", case.demographics.name)

    parts = ["\n\n=== PATIENT CASE DETAILS ===\n"]

    demo = case.demographics
    parts.append(
        f"<demographics>\n"
        f"name: {demo.name}\n"
        f"date_of_birth: {demo.date_of_birth}\n"
        f"sex: {demo.sex}\n"
        f"gender: {demo.gender}\n"
        f"background: {demo.background}\n"
        f"</demographics>\n"
    )

    if case.behavior:
        parts.append(f"<behavior>\n{case.behavior}\n</behavior>\n")

    parts.append(f"<chief_concern>\n{case.chief_concern}\n</chief_concern>\n")

    free_items = "\n".join(f"- {item}" for item in case.free_information)
    parts.append(
        f"<free_information>\n"
        f"Information you may volunteer or mention naturally:\n"
        f"{free_items}\n"
        f"</free_information>\n"
    )

    locked_items = "\n".join(f"- {item}" for item in case.locked_information)
    parts.append(
        f"<locked_information>\n"
        f"Information to ONLY reveal when the student asks appropriate, specific questions:\n"
        f"{locked_items}\n"
        f"</locked_information>"
    )

    return base + "\n".join(parts)
