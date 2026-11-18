"""LAB 09 — Option A: NER sang PERSON/ORG/LOCATION; gold IE 30 câu."""
from pathlib import Path
import json,re
import torch
from transformers import BertTokenizerFast
from baseline_ner import CACHE,PretrainedTokenClassifier,assets,predict,spans,score,configure

# Câu kiểu tin tức tự biên soạn, không phải bản tin về sự kiện thực tế.
# Tên entity và loại được gán thủ công trước khi chạy model.
GOLD_FACTS=[
('Barack Obama','United Nations','New York'),('Angela Merkel','European Union','Berlin'),
('Emmanuel Macron','UNESCO','Paris'),('Joe Biden','NATO','Brussels'),('Rishi Sunak','BBC','London'),
('Justin Trudeau','Reuters','Ottawa'),('Tim Cook','Apple','California'),('Satya Nadella','Microsoft','Seattle'),
('Sundar Pichai','Google','London'),('Elon Musk','Tesla','Texas'),('Jeff Bezos','Amazon','Washington'),
('Mark Zuckerberg','Meta','New York'),('Jensen Huang','Nvidia','Taiwan'),('Bill Gates','Microsoft','Paris'),
('Warren Buffett','Berkshire Hathaway','Nebraska'),('Serena Williams','Nike','New York'),
('Roger Federer','UNICEF','Geneva'),('Lionel Messi','FIFA','Argentina'),('Cristiano Ronaldo','FIFA','Portugal'),
('Taylor Swift','Sony','Tokyo'),('Adele','BBC','London'),('Beyonce','UNICEF','Houston'),
('Malala Yousafzai','United Nations','Pakistan'),('Greta Thunberg','Greenpeace','Sweden'),
('Nelson Mandela','African National Congress','South Africa'),('Kofi Annan','United Nations','Ghana'),
('Ban Ki-moon','United Nations','Seoul'),('Jacinda Ardern','Labour Party','Wellington'),
('Anthony Albanese','Labor Party','Canberra'),('Narendra Modi','World Bank','India')]
TEMPLATES=[
'{p} spoke at {o} in {l}.',
'In {l}, {p} met representatives of {o}.',
'{o} invited {p} to a conference in {l}.',
'A reporter in {l} asked {p} about {o}.',
'{p} discussed the work of {o} during a visit to {l}.']

def words(text):return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]",text)

def gold_corpus():
    rows=[]
    for i,(person,org,location) in enumerate(GOLD_FACTS):
        text=TEMPLATES[i%len(TEMPLATES)].format(p=person,o=org,l=location)
        tokens=words(text);tags=['O']*len(tokens)
        for surface,kind in [(person,'PER'),(org,'ORG'),(location,'LOC')]:
            target=words(surface)
            matches=[j for j in range(len(tokens)-len(target)+1) if tokens[j:j+len(target)]==target]
            assert len(matches)==1
            a=matches[0];tags[a]='B-'+kind
            for j in range(a+1,a+len(target)):tags[j]='I-'+kind
        rows.append({'text':text,'words':tokens,'tags':tags})
    return rows

def load():
    configure();path=assets();tokenizer=BertTokenizerFast.from_pretrained(path,do_lower_case=True)
    model=PretrainedTokenClassifier(pretrained=False)
    model.load_state_dict(torch.load(CACHE/'ner.pt',map_location='cpu',weights_only=True))
    return model.eval(),tokenizer

def structured(tokens,tags):
    mapping={'PER':'PERSON','ORG':'ORG','LOC':'LOCATION'}
    result={k:[] for k in mapping.values()}
    for a,b,t in spans(tags):
        if t in mapping:result[mapping[t]].append(' '.join(tokens[a:b]))
    return result

def extract(text,model=None,tokenizer=None):
    if model is None:model,tokenizer=load()
    tokens=words(text)
    if not tokens:return {'PERSON':[],'ORG':[],'LOCATION':[]}
    tags=predict(model,[{'words':tokens,'tags':['O']*len(tokens)}],tokenizer)[0]
    return structured(tokens,tags)

def evaluate_ie():
    model,tokenizer=load();rows=gold_corpus();pred=predict(model,rows,tokenizer)
    # MISC không thuộc schema IE; vẫn lưu dự đoán NER đầy đủ để đối chiếu.
    filtered=[[t if t=='O' or t[2:] in {'PER','ORG','LOC'} else 'O' for t in seq] for seq in pred]
    result={'sentences':len(rows),'gold_entities':90,'metrics':score([r['tags'] for r in rows],filtered),
            'examples':[{'Text':r['text'],'Gold':structured(r['words'],r['tags']),
                         'Prediction':structured(r['words'],p),'gold_tags':r['tags'],'pred_tags':p}
                        for r,p in zip(rows,filtered)]}
    (CACHE/'ie_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--text');args=parser.parse_args()
    print(json.dumps(extract(args.text) if args.text else evaluate_ie(),ensure_ascii=False,indent=2))
