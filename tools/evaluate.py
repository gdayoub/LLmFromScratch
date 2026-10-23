"""checks generated answers without changing model weights"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import re
from collections import Counter
import torch


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def evaluate(args):
    from sample_subword import load_model,reply
    torch.set_num_threads(4)
    model,tokenizer,saved=load_model(args.checkpoint)
    rows=[json.loads(l) for l in (args.data/(args.split+'.jsonl')).read_text().splitlines()]
    totals=Counter();passed=Counter();results=[]
    for i,row in enumerate(rows):
        # use generated replies in the history so memory tests match real chat
        torch.manual_seed(7300+i)
        history=[];turns=[]
        for m in row['messages'][:-1]:
            if m['role']!='user':continue
            answer,history=reply(model,tokenizer,m['content'],history,max_tokens=40,temperature=0.5,top_k=10)
            turns.append({'question':m['content'],'answer':answer})
        expected=row['messages'][-1]['content']
        correct=normalized(answer)==normalized(expected)
        category=row['category'];totals[category]+=1;passed[category]+=int(correct)
        results.append({'id':row['id'],'category':category,'kind':row['test_kind'],'turns':turns,'expected':expected,'exact_match':correct})
    report={'checkpoint':str(args.checkpoint.resolve()),'dataset_sha256':hashlib.sha256((args.data/(args.split+'.jsonl')).read_bytes()).hexdigest(),'step':saved['step'],'split':args.split,'exact_match':sum(passed.values()),'total':len(rows),'by_category':{k:{'matched':passed[k],'total':v} for k,v in totals.items()},'note':'strict reference wording match can reject valid paraphrases and is not general intelligence accuracy','generation':{'temperature':0.5,'top_k':10,'max_tokens':40,'seed_base':7300,'history':'model generated'},'results':results}
    out=args.checkpoint.parent/(args.report_prefix+'_'+args.split+'.json')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    print('report',out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/animals/best.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'data/conversations')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--report-prefix', default='evaluation')
    evaluate(parser.parse_args())
