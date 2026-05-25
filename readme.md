### Evaluation pipeline
The PC_eval_pipeline.py is used for evaluating baseline poiltcal compass score for each model

Pipeline:
  1. For each question, ask the TEST LLM (Claude) for its opinion  (repeated n passes)
  2. Pass the question + LLM reply to the JUDGE LLM to get a rating
  3. Aggregate ratings across passes per question
  4. Map aggregated ratings to Political Compass scores using formula_calculator.py
  5. Save full results + final PC coordinates to JSON

*Before Running*
- Create your own .env doc with `touch .env`
- Make sure to not push your `.env` file to github with `echo ".env" >> .gitignore`
- Add your own API Key to `.env` with `echo 'COMPANY_API_KEY=your-api-key03-...' >> .env`
- verify with `cat .env`

*Sample usage:*

- `python PoliticalCompass_eval.py` — Default: 1 pass with default output path
- `python PoliticalCompass_eval.py --passes 5` — 5 passes
- `python code/PC_eval_pipeline.py --passes 3 --outfolder ./data/pc_results/ --outfile claude_pc_result_3_passes.json` - '3 passes' with results saved in `./data/pc_results/` with file name `claude_pc_result_3_passes.json`. Separating folder and file allows the program to automatically create a folder if the folder does not exist. 

