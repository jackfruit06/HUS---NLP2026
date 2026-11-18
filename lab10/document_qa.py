"""LAB 10 — Document QA với context được cung cấp trực tiếp; không retrieval."""
import json
from qa_model import (CACHE, BASE_LENGTH, assets, load_model, encode_rows, infer,
                      BertTokenizerFast)

def answer_question(question,context,model=None,tokenizer=None,max_length=BASE_LENGTH):
    if not question.strip() or not context.strip():
        raise ValueError('Question và context phải có nội dung.')
    if tokenizer is None:
        tokenizer=BertTokenizerFast.from_pretrained(assets(),do_lower_case=True)
    if model is None:
        if not (CACHE/f'qa-T{max_length}.pt').exists():
            raise FileNotFoundError('Chạy qa_model.py hoặc thí nghiệm fine-tuning trước.')
        model=load_model(max_length)
    row={'id':'document-question','question':question,'context':context,'answers':[]}
    features=encode_rows(tokenizer,[row],max_length)
    predictions,_=infer(model,features,[row])
    result=predictions[0]
    return {'Question':question,'Answer':result['answer'],'Start character':result['start_char'],
            'End character (exclusive)':result['end_char'],'Windows':len(features),
            'Span score':result['score']}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question',required=True)
    parser.add_argument('--context',required=True)
    args=parser.parse_args()
    print(json.dumps(answer_question(args.question,args.context),ensure_ascii=False,indent=2))
