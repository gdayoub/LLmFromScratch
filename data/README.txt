ANIMAL CONVERSATION DATA

Open conversations/train.jsonl to see readable questions and answers
2870 practice examples include earlier basic skills and the animal curriculum
29 fresh validation conversations select the checkpoint
33 fresh final test conversations are reserved until selection is finished

28 animal fact records are readable in sources/animal_facts.json
They cover groups sounds habitats and distinctive features
Some examples compare animals or ask a follow up about the previous animal
Questions about dangerous animals distinguish disease related human deaths
from physical strength or danger in an individual encounter
Source notes include CDC and Smithsonian references

Description reading examples are clearly fictional
Names foods and homes vary and some names and food home combinations are held out
These check whether the model uses information supplied in the question
They are not claims about real animals and are not broad intelligence tests

The chat program generates text from weights and does not read this answer file
Training still predicts the next token rather than installing a lookup table
There is no guarantee that new phrasing will work or that all answers are correct

english/ keeps English story text and sources/ keeps source records
artifacts/animals contains the current machine cache and tokenizer
artifacts/basic_improved_readable preserves the previous readable dataset
Only three tools remain active: prepare_data.py check_model.py evaluate.py
