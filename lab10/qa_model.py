"""LAB 10 — SQuAD alignment, start/end head, fine-tuning và EM/F1."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import json,re,string,random,hashlib,copy,time,collections,platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel,BertConfig,BertTokenizerFast

LAB=Path(__file__).resolve().parent
CACHE=LAB.parent/'.lab10_cache'
MODEL_ID='prajjwal1/bert-tiny'
REVISION='6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
SEED,BATCH_SIZE,EPOCHS,LR=42,16,3,0.0003
BASE_LENGTH,STRIDE,MAX_QUESTION,MAX_ANSWER=256,64,48,64

def configure():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)

def assets():
    import requests
    CACHE.mkdir(exist_ok=True);model=CACHE/'bert-tiny';model.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    for name in ['config.json','vocab.txt','pytorch_model.bin']:
        path=model/name
        if not path.exists():
            old=LAB.parent/'.lab9_cache/bert-tiny'/name
            if old.exists():path.write_bytes(old.read_bytes())
            else:
                r=session.get(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}',timeout=180)
                r.raise_for_status();path.write_bytes(r.content)
    for split in ['train','dev']:
        path=CACHE/f'{split}-v1.1.json'
        if not path.exists():
            r=session.get(f'https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v1.1.json',timeout=180)
            r.raise_for_status();path.write_bytes(r.content)
    return model

def fingerprint(values):return hashlib.sha256(json.dumps(values,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()

def prepare_data():
    assets()
    if (CACHE/'splits.json').exists():return json.loads((CACHE/'splits.json').read_text(encoding='utf-8'))
    def flatten(split):
        out=[]
        raw=json.loads((CACHE/f'{split}-v1.1.json').read_text(encoding='utf-8'))
        for article in raw['data']:
            for para in article['paragraphs']:
                for qa in para['qas']:
                    answers=qa['answers']
                    assert all(para['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in answers)
                    out.append({'id':qa['id'],'title':article['title'],'question':qa['question'],
                                'context':para['context'],'answers':answers})
        return out
    train,dev=flatten('train'),flatten('dev');rng=np.random.default_rng(SEED)
    train=[train[i] for i in rng.choice(len(train),3000,replace=False)]
    contexts=sorted({r['context'] for r in dev});order=rng.permutation(len(contexts))
    valid_contexts={contexts[i] for i in order[:len(contexts)//3]}
    validation=[r for r in dev if r['context'] in valid_contexts]
    test=[r for r in dev if r['context'] not in valid_contexts]
    validation=[validation[i] for i in rng.choice(len(validation),300,replace=False)]
    test=[test[i] for i in rng.choice(len(test),600,replace=False)]
    data={'train':train,'validation':validation,'test':test}
    sets=[{r['context'] for r in rows} for rows in data.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    (CACHE/'splits.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    return data

def normalize(text):
    text=text.lower();text=''.join(c for c in text if c not in string.punctuation)
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())

def answer_scores(prediction,gold):
    p,g=normalize(prediction).split(),normalize(gold).split()
    em=float(normalize(prediction)==normalize(gold))
    if not p or not g:return em,float(p==g)
    overlap=sum((collections.Counter(p)&collections.Counter(g)).values())
    return em,2*overlap/(len(p)+len(g))

def metrics(rows,predictions):
    em=[];f1=[]
    for row,pred in zip(rows,predictions):
        values=[answer_scores(pred['answer'],a['text']) for a in row['answers']]
        em.append(max(v[0] for v in values));f1.append(max(v[1] for v in values))
    return {'EM':100*float(np.mean(em)),'F1':100*float(np.mean(f1))}

def token_span(offsets,sequence_ids,answer_start,answer_end):
    context=[i for i,s in enumerate(sequence_ids) if s==1 and offsets[i][1]>offsets[i][0]]
    if not context or offsets[context[0]][0]>answer_start or offsets[context[-1]][1]<answer_end:return 0,0
    starts=[i for i in context if offsets[i][0]<=answer_start<offsets[i][1]]
    ends=[i for i in context if offsets[i][0]<answer_end<=offsets[i][1]]
    return (starts[0],ends[-1]) if starts and ends else (0,0)

def encode_rows(tokenizer,rows,max_length=BASE_LENGTH,stride=STRIDE):
    questions=[tokenizer.decode(tokenizer(r['question'],add_special_tokens=False,truncation=True,max_length=MAX_QUESTION)['input_ids']) for r in rows]
    batch=tokenizer(questions,[r['context'] for r in rows],truncation='only_second',max_length=max_length,
                    stride=stride,return_overflowing_tokens=True,return_offsets_mapping=True,padding=False)
    features=[]
    for k,sid in enumerate(batch['overflow_to_sample_mapping']):
        offsets=batch['offset_mapping'][k];seq=batch.sequence_ids(k)
        positions=[token_span(offsets,seq,a['answer_start'],a['answer_start']+len(a['text'])) for a in rows[sid]['answers']]
        start,end=next((pair for pair in positions if pair!=(0,0)),(0,0))
        features.append({'sample':int(sid),'id':rows[sid]['id'],'input_ids':batch['input_ids'][k],
            'attention_mask':batch['attention_mask'][k],'token_type_ids':batch['token_type_ids'][k],
            'offset_mapping':[list(pair) for pair in offsets],'sequence_ids':seq,'start_position':start,'end_position':end})
    return features

def collate(features):
    length=max(len(f['input_ids']) for f in features)
    result={k:torch.tensor([f[k]+[0]*(length-len(f[k])) for f in features],dtype=torch.long)
            for k in ['input_ids','attention_mask','token_type_ids']}
    result['start_positions']=torch.tensor([f['start_position'] for f in features],dtype=torch.long)
    result['end_positions']=torch.tensor([f['end_position'] for f in features],dtype=torch.long)
    return result

class PretrainedSpanQA(nn.Module):
    """Input IDs -> encoder -> token representations -> start/end logits."""
    def __init__(self,pretrained=True):
        super().__init__();path=CACHE/'bert-tiny'
        config=BertConfig.from_pretrained(path);config._attn_implementation='eager'
        self.encoder=BertModel.from_pretrained(path,config=config,add_pooling_layer=False) if pretrained else BertModel(config,add_pooling_layer=False)
        self.dropout=nn.Dropout(.1);self.span_classifier=nn.Linear(config.hidden_size,2)
    def forward(self,input_ids,attention_mask,token_type_ids):
        contextual_representations=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
        span_logits=self.span_classifier(self.dropout(contextual_representations))
        start_logits,end_logits=span_logits.unbind(-1)
        return start_logits,end_logits

def qa_loss(start_logits,end_logits,start_positions,end_positions):
    return (nn.functional.cross_entropy(start_logits,start_positions)+nn.functional.cross_entropy(end_logits,end_positions))/2

@torch.no_grad()
def infer(model,features,rows):
    model.eval();best=[None]*len(rows);total_loss=0.;count=0;offset=0
    for batch in DataLoader(features,batch_size=BATCH_SIZE,collate_fn=collate):
        starts,ends=model(**{k:batch[k] for k in ['input_ids','attention_mask','token_type_ids']})
        loss=qa_loss(starts,ends,batch['start_positions'],batch['end_positions'])
        total_loss+=loss.item()*len(starts);count+=len(starts)
        for j in range(len(starts)):
            feature=features[offset];offset+=1;sid=feature['sample']
            valid=np.array([i for i,s in enumerate(feature['sequence_ids']) if s==1])
            if not len(valid):continue
            sl,el=starts[j].numpy(),ends[j].numpy()
            si=valid[np.argsort(-sl[valid],kind='stable')[:20]]
            ei=valid[np.argsort(-el[valid],kind='stable')[:20]]
            for s in si:
                for t in ei:
                    if t<s or t-s+1>MAX_ANSWER:continue
                    value=float(sl[s]+el[t]);a,b=feature['offset_mapping'][s][0],feature['offset_mapping'][t][1]
                    if b<=a:continue
                    if best[sid] is None or value>best[sid]['score']:
                        best[sid]={'id':rows[sid]['id'],'answer':rows[sid]['context'][a:b],
                            'start_char':a,'end_char':b,'start_token':int(s),'end_token':int(t),'score':value,'window':offset-1}
    assert all(x is not None for x in best)
    return best,total_loss/count

def lexical_baseline(row):
    """Chọn câu nhiều keyword overlap nhất, trả phrase sau is/was hoặc tên/số."""
    context=row['context'];question=row['question'];q=set(normalize(question).split())
    candidates=list(re.finditer(r'[^.!?]+(?:[.!?]|$)',context))
    if not candidates:return {'id':row['id'],'answer':'','start_char':0,'end_char':0}
    sent=max(candidates,key=lambda m:len(q&set(normalize(m.group()).split()))/(1+len(set(normalize(m.group()).split()))**.5))
    text=sent.group();patterns=[]
    if re.search(r'\b(when|year|how many|how much)\b',question.lower()):patterns.append(r'\b\d[\d,]*(?:\.\d+)?\b')
    if re.match(r'\s*who\b',question.lower()):patterns.append(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}')
    patterns.extend([r'\b(?:is|was|were|are)\s+([^,;.]+)',r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}'])
    for pattern in patterns:
        m=re.search(pattern,text)
        if m:
            a,b=m.span(1) if m.lastindex else m.span();a+=sent.start();b+=sent.start()
            return {'id':row['id'],'answer':context[a:b],'start_char':a,'end_char':b}
    a=sent.start()+len(text)-len(text.lstrip());b=min(a+len(text.strip()),a+60)
    return {'id':row['id'],'answer':context[a:b],'start_char':a,'end_char':b}

def load_model(length=BASE_LENGTH):
    configure();assets();model=PretrainedSpanQA(pretrained=False)
    model.load_state_dict(torch.load(CACHE/f'qa-T{length}.pt',map_location='cpu',weights_only=True))
    return model.eval()

def train_one(rows,tokenizer,length):
    configure();features={split:encode_rows(tokenizer,values,length) for split,values in rows.items()}
    model=PretrainedSpanQA();optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.01)
    train_loader=DataLoader(features['train'],batch_size=BATCH_SIZE,shuffle=True,
                            generator=torch.Generator().manual_seed(SEED),collate_fn=collate)
    history=[];best=-1;state=None;started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train();loss_sum=0.;count=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            sl,el=model(**{k:batch[k] for k in ['input_ids','attention_mask','token_type_ids']})
            loss=qa_loss(sl,el,batch['start_positions'],batch['end_positions'])
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=loss.item()*len(sl);count+=len(sl)
        predictions,vl=infer(model,features['validation'],rows['validation']);scores=metrics(rows['validation'],predictions)
        history.append({'Epoch':epoch,'Training loss':loss_sum/count,'Validation loss':vl,'Validation EM':scores['EM'],'Validation F1':scores['F1']})
        if scores['F1']>best:best=scores['F1'];state=copy.deepcopy(model.state_dict());best_epoch=epoch
        print('T=',length,history[-1],flush=True)
    elapsed=time.perf_counter()-started;model.load_state_dict(state);torch.save(state,CACHE/f'qa-T{length}.pt')
    predictions,test_loss=infer(model,features['test'],rows['test'])
    coverage=sum(any(f['sample']==i and f['start_position']!=0 for f in features['test']) for i in range(len(rows['test'])))
    return {'row':{'Model':'Fine-tuned Transformer','Max length':length,'EM':metrics(rows['test'],predictions)['EM'],
        'F1':metrics(rows['test'],predictions)['F1'],'Training time':elapsed,'Best epoch':best_epoch,
        'Train samples':len(rows['train']),'Validation samples':len(rows['validation']),'Test samples':len(rows['test']),
        'Train windows':len(features['train']),'Test windows':len(features['test']),'Answer-covered test questions':coverage},
        'history':history,'predictions':predictions,'test_loss':test_loss}

def run():
    configure();model_path=assets();data=prepare_data();tokenizer=BertTokenizerFast.from_pretrained(model_path,do_lower_case=True)
    baseline=[lexical_baseline(r) for r in data['test']]
    result={'baseline':{'row':{'Model':'Baseline',**metrics(data['test'],baseline)},'predictions':baseline},'runs':{}}
    for length in [128,256,384]:
        result['runs'][str(length)]=train_one(data,tokenizer,length)
        (CACHE/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        pd.DataFrame([r['row'] for r in result['runs'].values()]+[result['baseline']['row']]).to_csv(LAB/'results.csv',index=False)
    analysis=analyze_predictions(data,result)
    (CACHE/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def analyze_predictions(data,evidence):
    import sys
    qa=sys.modules[__name__]
    e=evidence
    rows=data['test'];pred=e['runs'][str(qa.BASE_LENGTH)]['predictions']

    def per_row(row,p):
     values=[qa.answer_scores(p['answer'],a['text']) for a in row['answers']]
     return max(x[0] for x in values),max(x[1] for x in values)

    partial=[];categories={'wrong start':[],'wrong end':[],'wrong span':[]};counts=collections.Counter()
    for row,p in zip(rows,pred):
     em,f1=per_row(row,p)
     pairs=[(a['answer_start'],a['answer_start']+len(a['text'])) for a in row['answers']]
     start_ok=any(p['start_char']==a for a,b in pairs);end_ok=any(p['end_char']==b for a,b in pairs)
     counts['# wrong start']+=int(not start_ok);counts['# wrong end']+=int(not end_ok)
     counts['# both boundaries wrong']+=int(not start_ok and not end_ok)
     if em==0 and f1>0 and len(partial)<5:
      partial.append({'Question':row['question'],'Gold':' | '.join(a['text'] for a in row['answers']),
                      'Prediction':p['answer'],'EM':em,'F1':f1,
                      'Explanation':'Có từ sau normalization trùng gold nhưng thừa/thiếu hoặc khác boundary nên EM=0.'})
     if em:continue
     kind='wrong start' if not start_ok and end_ok else 'wrong end' if start_ok and not end_ok else 'wrong span' if not start_ok and not end_ok else None
     if kind is None or len(categories[kind])>=5:continue
     best_gold=max(row['answers'],key=lambda a:qa.answer_scores(p['answer'],a['text'])[1])
     a,b=best_gold['answer_start'],best_gold['answer_start']+len(best_gold['text'])
     overlaps=max(a,p['start_char'])<min(b,p['end_char'])
     window_count=sum(1 for f in qa.encode_rows(qa.BertTokenizerFast.from_pretrained(qa.CACHE/'bert-tiny'),[row],qa.BASE_LENGTH))
     explanation=f"Gold chars [{a},{b}), pred [{p['start_char']},{p['end_char']}); "
     explanation+=('có phần giao nhau, phù hợp lỗi partial answer/boundary.' if overlaps else 'span nằm ở vùng khác context; có thể nhầm thông tin hoặc distractor, chưa xác định nguyên nhân.')
     explanation+=f' Context tạo {window_count} window; mọi window đều được xét.'
     categories[kind].append({'Question':row['question'],'Gold':best_gold['text'],'Prediction':p['answer'],
                              'Error type':kind,'Explanation':explanation,'id':row['id']})
    assert len(partial)==5 and all(len(v)==5 for v in categories.values()),{k:len(v) for k,v in categories.items()}

    length=[]
    for name,lo,hi in [('1',1,1),('2–3',2,3),('4–6',4,6),('7+',7,100000)]:
     indices=[i for i,r in enumerate(rows) if lo<=len(r['answers'][0]['text'].split())<=hi]
     length.append({'Answer length':name,'Samples':len(indices),**qa.metrics([rows[i] for i in indices],[pred[i] for i in indices])})

    stop={'what','which','when','where','who','how','is','was','were','are','did','do','does','of','in','to','for','and','on','by','with'}
    matching=[];match_examples=[]
    def overlap(row):
     q=set(qa.normalize(row['question']).split())-stop;c=set(qa.normalize(row['context']).split())
     return len(q&c)/max(1,len(q))
    for name,condition in [('High overlap (≥0.75)',lambda x:x>=.75),('Lower overlap (<0.75)',lambda x:x<.75)]:
     indices=[i for i,r in enumerate(rows) if condition(overlap(r))]
     matching.append({'Group':name,'Samples':len(indices),**qa.metrics([rows[i] for i in indices],[pred[i] for i in indices])})
     for i in indices[:3]:match_examples.append({'Group':name,'Question':rows[i]['question'],'Gold':rows[i]['answers'][0]['text'],
         'Prediction':pred[i]['answer'],'Overlap':overlap(rows[i])})

    model=qa.load_model();tokenizer=qa.BertTokenizerFast.from_pretrained(qa.CACHE/'bert-tiny')
    original=[];distractor=[];reordered=[];original_pred=[]
    for row,p in zip(rows[:80],pred[:80]):
     sentences=[m.group().strip() for m in re.finditer(r'[^.!?]+(?:[.!?]|$)',row['context'])]
     new_context=' '.join(reversed(sentences))
     if len(sentences)<2 or not all(a['text'] in new_context for a in row['answers']):continue
     original.append(row);original_pred.append(p)
     prefix='A separate note repeats the question: "'+row['question']+'" without giving its answer. '
     distractor.append({**row,'context':prefix+row['context'],
                        'answers':[{'text':a['text'],'answer_start':len(prefix)+a['answer_start']} for a in row['answers']]})
     reordered.append({**row,'context':new_context,'answers':[{'text':a['text'],'answer_start':new_context.index(a['text'])} for a in row['answers']]})
     if len(original)==50:break
    controls=[{'Condition':'Original','Samples':len(original),**qa.metrics(original,original_pred)}]
    control_predictions={}
    for name,subset in [('Question-word distractor',distractor),('Reversed sentences',reordered)]:
     features=qa.encode_rows(tokenizer,subset);out,_=qa.infer(model,features,subset)
     controls.append({'Condition':name,'Samples':len(subset),**qa.metrics(subset,out)})
     control_predictions[name]=out

    from document_qa import answer_question
    faq_context=('The university library opens at 8 a.m. on weekdays. Students can borrow up to five books. '
                 'Books must be returned within fourteen days. The help desk is on the second floor.')
    document=[answer_question(question,faq_context,model,tokenizer) for question in [
        'How many books can students borrow?','Where is the help desk?','When must books be returned?']]
    analysis={'partial':partial,'errors':sum(categories.values(),[]),'boundary_counts':dict(counts),'length':length,
              'matching':matching,'matching_examples':match_examples,'controls':controls,
              'control_ids':[r['id'] for r in original],'control_predictions':control_predictions,
              'document_context':faq_context,'document_examples':document}
    analysis.update(question_matching_pairs(model,tokenizer))
    return analysis

def question_matching_pairs(model,tokenizer):
    """Sáu cặp câu hỏi cùng context/gold, thay wording literal bằng paraphrase."""
    cases=[
        ('The capital of France is Paris.','What city is the capital of France?',
         'Where is the French national government based?','Paris'),
        ('Marie Curie discovered radium in 1898.','What did Marie Curie discover?',
         'Which element was identified by the scientist named Marie Curie?','radium'),
        ('The conference will begin on Monday.','When will the conference begin?',
         'On which day is the meeting scheduled to start?','Monday'),
        ('The library closes at six in the evening.','When does the library close?',
         'What time does the book-lending facility shut its doors?','six in the evening'),
        ('Students must return books within fourteen days.','Within how many days must students return books?',
         'What is the loan period for borrowed books?','fourteen days'),
        ('The committee selected Alice as its chair.','Who did the committee select as its chair?',
         'Who was appointed to lead the panel?','Alice')]
    records=[];reports=[]
    for kind,index in [('Lexical match',1),('Paraphrase',2)]:
        rows=[{'id':f'pair-{i}-{kind}','context':c[0],'question':c[index],
               'answers':[{'text':c[3],'answer_start':c[0].index(c[3])}]} for i,c in enumerate(cases)]
        predictions,_=infer(model,encode_rows(tokenizer,rows),rows)
        reports.append({'Question type':kind,'Samples':len(rows),**metrics(rows,predictions)})
        records.extend({'Pair':i,'Question type':kind,'Question':r['question'],'Context':r['context'],
                        'Gold':r['answers'][0]['text'],'Prediction':p['answer']} for i,(r,p) in enumerate(zip(rows,predictions)))
    return {'matching_pairs':records,'matching_pair_metrics':reports}

if __name__=='__main__':run()
