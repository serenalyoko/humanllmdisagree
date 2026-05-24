### Evaluation pipeline
The PC_eval_pipeline.py is used for evaluating baseline poiltcal compass score for each model
Pipeline:
  1. For each question, ask the TEST LLM (Claude) for its opinion  (repeated n passes)
  2. Pass the question + LLM reply to the JUDGE LLM to get a rating
  3. Aggregate ratings across passes per question
  4. Map aggregated ratings to Political Compass scores using formula_calculator.py
  5. Save full results + final PC coordinates to JSON

sample usage:
    python PoliticalCompass_eval.py            # 1 pass (default)
    python PoliticalCompass_eval.py --passes 5 # 5 passes