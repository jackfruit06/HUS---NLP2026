"""LAB 09: token features, alignment, BERT token classifier và entity metrics."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import sys, json, re, random, time, copy, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from transformers import BertModel, BertConfig, BertTokenizerFast

LAB=Path(__file__).resolve().parent
CACHE=LAB.parent/'.lab9_cache'
sys.path.insert(0,str(CACHE/'deps'))
MODEL_ID='prajjwal1/bert-tiny'
MODEL_REVISION='6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
TAGS=['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC']
SEED,MAX_LENGTH,BATCH_SIZE,EPOCHS,LR=42,128,32,4,0.0003

def configure():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)

def assets():
    import requests
    CACHE.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    model=CACHE/'bert-tiny';model.mkdir(exist_ok=True)
    for name in ['config.json','vocab.txt','pytorch_model.bin']:
        path=model/name
        if not path.exists():
            old=LAB.parent/'.lab08_cache/bert-tiny'/name
            if old.exists():path.write_bytes(old.read_bytes())
            else:
                r=session.get(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}',timeout=180)
                r.raise_for_status();path.write_bytes(r.content)
    if not (CACHE/'data.json').exists():
        import pyarrow.parquet as pq
        meta=session.get('https://huggingface.co/api/datasets/tomaarsen/conll2003',timeout=60)
        meta.raise_for_status();revision=meta.json()['sha']
        data={}
        for split in ['train','validation','test']:
            r=session.get(f'https://huggingface.co/datasets/tomaarsen/conll2003/resolve/{revision}/data/{split}-00000-of-00001.parquet',timeout=180)
            r.raise_for_status();p=CACHE/f'{split}.parquet';p.write_bytes(r.content)
            table=pq.read_table(p)
            features=json.loads(table.schema.metadata[b'huggingface'])['info']['features']
            names=features['ner_tags']['feature']['names']
            assert names==TAGS,names
            data[split]=[{'words':row['tokens'],'tags':bio2([TAGS[i] for i in row['ner_tags']])} for row in table.to_pylist()]
        (CACHE/'data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
        (CACHE/'dataset_info.json').write_text(json.dumps({'source':'tomaarsen/conll2003','revision':revision,
            'sha256':{s:hashlib.sha256((CACHE/f'{s}.parquet').read_bytes()).hexdigest() for s in data}},indent=2),encoding='utf-8')
    return model

def bio2(tags):
    result=[];previous='O'
    for tag in tags:
        if tag.startswith('I-') and (previous=='O' or previous[2:]!=tag[2:]):tag='B-'+tag[2:]
        result.append(tag);previous=tag
    return result

def spans(tags):
    """CoNLL-style chunks: I-X đầu chuỗi/sau loại khác bắt đầu chunk X."""
    result=[];start=None;kind=None
    for i,tag in enumerate(list(tags)+['O']):
        new_kind=tag[2:] if tag!='O' else None
        if kind is not None and (tag=='O' or tag.startswith('B-') or new_kind!=kind):
            result.append((start,i,kind));start=None;kind=None
        if tag!='O' and kind is None:start=i;kind=new_kind
    return result

def entity_sets(sequences):
    return {(i,a,b,t) for i,tags in enumerate(sequences) for a,b,t in spans(tags)}

def score(gold,pred,kind=None,length=None):
    g,p=entity_sets(gold),entity_sets(pred)
    def keep(x):return (kind is None or x[3]==kind) and (length is None or min(x[2]-x[1],4)==length)
    g,p={x for x in g if keep(x)},{x for x in p if keep(x)}
    tp=len(g&p);fp=len(p-g);fn=len(g-p)
    precision=tp/len(p) if p else 0.;recall=tp/len(g) if g else 0.
    return {'Precision':precision,'Recall':recall,'F1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,'TP':tp,'FP':fp,'FN':fn,'Gold entities':len(g)}

def features(words,i):
    w=words[i]
    return {'word':w.lower(),'prefix2':w[:2].lower(),'suffix3':w[-3:].lower(),
            'title':w.istitle(),'upper':w.isupper(),'digit':w.isdigit(),
            'shape':re.sub('[a-z]','x',re.sub('[A-Z]','X',w)),
            'previous':words[i-1].lower() if i else '<BOS>',
            'next':words[i+1].lower() if i+1<len(words) else '<EOS>'}

class TokenFeatureBaseline:
    def fit(self,rows):
        self.vectorizer=DictVectorizer(dtype=np.float32)
        x=self.vectorizer.fit_transform([features(r['words'],i) for r in rows for i in range(len(r['words']))])
        y=[tag for r in rows for tag in r['tags']]
        self.classifier=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=30,tol=1e-4,random_state=SEED)
        self.classifier.fit(x,y);return self
    def predict(self,rows):
        x=self.vectorizer.transform([features(r['words'],i) for r in rows for i in range(len(r['words']))])
        labels=self.classifier.predict(x).tolist();out=[];offset=0
        for row in rows:
            n=len(row['words']);out.append(labels[offset:offset+n]);offset+=n
        return out

def align(tokenizer,words,tags):
    encoded=tokenizer(words,is_split_into_words=True,add_special_tokens=True)
    ids=encoded.word_ids();previous=None;labels=[]
    for wid in ids:
        labels.append(TAGS.index(tags[wid]) if wid is not None and wid!=previous else -100)
        previous=wid
    return encoded,labels

def chunks(tokenizer,rows):
    out=[]
    for sid,row in enumerate(rows):
        enc=tokenizer(row['words'],is_split_into_words=True,add_special_tokens=False)
        counts=np.bincount(enc.word_ids(),minlength=len(row['words']))
        start=0;used=0
        for j,count in enumerate(counts):
            if used+count>MAX_LENGTH-2 and j>start:
                out.append((sid,start,row['words'][start:j],row['tags'][start:j]));start=j;used=0
            if count>MAX_LENGTH-2:raise ValueError('Một word vượt ngân sách subword.')
            used+=count
        if start<len(counts):out.append((sid,start,row['words'][start:],row['tags'][start:]))
    result=[]
    for sid,start,words,tags in out:
        enc,labels=align(tokenizer,words,tags)
        assert len(enc['input_ids'])<=MAX_LENGTH
        result.append({'sid':sid,'start':start,'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],'labels':labels})
    return result

def collate(items):
    length=max(len(x['input_ids']) for x in items)
    return {key:torch.tensor([x[key]+[pad]*(length-len(x[key])) for x in items],dtype=torch.long)
            for key,pad in [('input_ids',0),('attention_mask',0),('labels',-100)]}

class PretrainedTokenClassifier(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();path=CACHE/'bert-tiny'
        config=BertConfig.from_pretrained(path);config._attn_implementation='eager'
        self.encoder=BertModel.from_pretrained(path,config=config,add_pooling_layer=False) if pretrained else BertModel(config,add_pooling_layer=False)
        self.dropout=nn.Dropout(.1);self.token_classifier=nn.Linear(config.hidden_size,len(TAGS))
    def forward(self,input_ids,attention_mask):
        contextual_representations=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        token_logits=self.token_classifier(self.dropout(contextual_representations))
        return token_logits

@torch.no_grad()
def predict(model,rows,tokenizer):
    model.eval();parts=chunks(tokenizer,rows);out=[[] for _ in rows];offset=0
    for batch in DataLoader(parts,batch_size=BATCH_SIZE,collate_fn=collate):
        logits=model(batch['input_ids'],batch['attention_mask']);labels=logits.argmax(-1)
        for j in range(len(labels)):
            part=parts[offset];offset+=1
            out[part['sid']].extend(TAGS[k] for k in labels[j][batch['labels'][j]!=-100].tolist())
    assert all(len(p)==len(r['words']) for p,r in zip(out,rows))
    return out

def analyze_predictions(data,gold,predictions):
    from collections import Counter,defaultdict
    gs,ps=entity_sets(gold),entity_sets(predictions)
    counts=Counter(' '.join(r['words'][a:b]).lower() for r in data['train'] for a,b,t in spans(r['tags']))
    rarity=[]
    for name,condition in [('Unseen (0)',lambda n:n==0),('Rare (1–2)',lambda n:1<=n<=2),('Frequent (3+)',lambda n:n>=3)]:
        group=[x for x in gs if condition(counts[' '.join(data['test'][x[0]]['words'][x[1]:x[2]]).lower()])]
        hits=sum(x in ps for x in group)
        rarity.append({'Train frequency':name,'Gold entities':len(group),'Exact matches':hits,'Recall':hits/len(group) if group else 0.})
    boundary=[]
    for p in sorted(ps-gs):
        for g in sorted(gs-ps):
            if p[0]==g[0] and p[3]==g[3] and max(p[1],g[1])<min(p[2],g[2]) and p[1:3]!=g[1:3]:
                boundary.append((p,g));break
    boundary=boundary[:5];usedp={p for p,g in boundary};usedg={g for p,g in boundary}
    fp=[p for p in sorted(ps-gs) if p not in usedp][:5];fn=[g for g in sorted(gs-ps) if g not in usedg][:5]
    errors=[]
    for category,items in [('FP',fp),('FN',fn),('Boundary',boundary)]:
        for j,item in enumerate(items,1):
            p,g=item if category=='Boundary' else (item,None) if category=='FP' else (None,item)
            anchor=p or g;sid=anchor[0];words=data['test'][sid]['words']
            overlap=[x for x in (gs if p else ps) if x[0]==sid and max(x[1],anchor[1])<min(x[2],anchor[2])]
            if category!='Boundary':
                if p:g=overlap[0] if overlap else None
                else:p=overlap[0] if overlap else None
            def form(x):return ' '.join(words[x[1]:x[2]])+' / '+x[3] if x else '∅'
            kind='wrong boundary' if category=='Boundary' else 'wrong entity type' if p and g and p[3]!=g[3] else 'spurious entity' if category=='FP' else 'missing entity'
            frequency=counts[' '.join(words[anchor[1]:anchor[2]]).lower()]
            explanation=(f'Gold và prediction khác đầu mút: gold [{g[1]},{g[2]}), pred [{p[1]},{p[2]}).' if category=='Boundary' else
                f"{'Prediction không có span khớp gold' if category=='FP' else 'Gold không có span khớp prediction'}; surface xuất hiện {frequency} lần trong train. "+
                ('Có entity chồng lấn khác loại, gợi ý nhầm loại theo context.' if p and g and p[3]!=g[3] else 'Cần xét ngữ cảnh câu; tần suất thấp là giả thuyết, chưa xác định nguyên nhân.'))
            errors.append({'ID':category+str(j),'Sentence':sid,'Text':' '.join(words),'Gold':form(g),'Prediction':form(p),'Error type':kind,'Explanation':explanation})
    contexts=defaultdict(dict)
    for sid,row in enumerate(data['test']):
        for a,b,t in spans(row['tags']):contexts[' '.join(row['words'][a:b])].setdefault(t,(sid,a,b))
    examples=[]
    for surface,types in sorted(contexts.items(),key=lambda x:(-len(x[1]),x[0])):
        if len(types)>1:
            for kind,(sid,a,b) in list(types.items())[:2]:
                examples.append({'Surface':surface,'Gold type':kind,'Sentence':sid,'Text':' '.join(data['test'][sid]['words']),
                                 'Predicted spans':str([(s,t,k) for s,t,k in spans(predictions[sid]) if max(s,a)<min(t,b)])})
            if len({r['Surface'] for r in examples})==3:break
    return {'errors':errors,'contexts':examples,'rarity':rarity}

def run():
    configure();path=assets();tokenizer=BertTokenizerFast.from_pretrained(path,do_lower_case=True)
    data=json.loads((CACHE/'data.json').read_text(encoding='utf-8'))
    gold=[r['tags'] for r in data['test']]
    started=time.perf_counter();baseline=TokenFeatureBaseline().fit(data['train']);elapsed=time.perf_counter()-started
    bp=baseline.predict(data['test']);rows=[{'Model':'Baseline','Training time (s)':elapsed,**score(gold,bp)}]
    print('Baseline',rows[-1],flush=True)
    model=PretrainedTokenClassifier();train=chunks(tokenizer,data['train'])
    loader=DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(SEED),collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.01)
    history=[];best=-1;state=None;started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train();loss_sum=0.;count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch['input_ids'],batch['attention_mask'])
            loss=nn.functional.cross_entropy(logits.reshape(-1,len(TAGS)),batch['labels'].reshape(-1),ignore_index=-100)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            n=int((batch['labels']!=-100).sum());loss_sum+=loss.item()*n;count+=n
        vp=predict(model,data['validation'],tokenizer);metrics=score([r['tags'] for r in data['validation']],vp)
        history.append({'Epoch':epoch,'Train loss':loss_sum/count,'Validation F1':metrics['F1']})
        if metrics['F1']>best:best=metrics['F1'];state=copy.deepcopy(model.state_dict())
        print(history[-1],flush=True)
    elapsed=time.perf_counter()-started;model.load_state_dict(state);torch.save(state,CACHE/'ner.pt')
    tp=predict(model,data['test'],tokenizer)
    rows.append({'Model':'Pretrained Transformer','Training time (s)':elapsed,**score(gold,tp)})
    pd.DataFrame(rows).to_csv(LAB/'results.csv',index=False)
    types=[{'Entity':t,**score(gold,tp,kind=t)} for t in ['PER','ORG','LOC','MISC']]
    pd.DataFrame(types).to_csv(LAB/'entity_metrics.csv',index=False)
    evidence={'results':rows,'type_metrics':types,'history':history,'gold':gold,'baseline':bp,'transformer':tp,
              'length':[{'Entity length':str(n) if n<4 else '4+',**score(gold,tp,length=n)} for n in range(1,5)],
              'configuration':{'seed':SEED,'epochs':EPOCHS,'lr':LR,'batch_size':BATCH_SIZE,'max_length':MAX_LENGTH,'train_chunks':len(train)}}
    (CACHE/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    return evidence

if __name__=='__main__':run()
