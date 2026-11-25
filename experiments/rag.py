"""Frozen, authored DEV/TEST cases. Reports lookup and conceptual QA separately.

This is a local educational evaluation, not a clinical or public QA benchmark.
The case file is not used by retrieval/generation or to fit their parameters.
"""
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from .common import ROOT,sha,write_json,finish_run,check_run
from demo.domain_knowledge import retrieve_domain,corpus
from demo.knowledge import normalize,DOCUMENTS


def token_f1(actual,reference):
    a=Counter(normalize(actual).split());b=Counter(normalize(reference).split())
    common=sum((a&b).values())
    if not common:return 0.
    p=common/sum(a.values());r=common/sum(b.values())
    return 2*p*r/(p+r)


def rouge_l(actual,reference):
    a=normalize(actual).split();b=normalize(reference).split();previous=[0]*(len(b)+1)
    for word in a:
        current=[0]
        for j,other in enumerate(b):current.append(previous[j]+1 if word==other else max(current[-1],previous[j+1]))
        previous=current
    if not a or not b:return 0.
    return 2*previous[-1]/(len(a)+len(b))


def run(generation=True):
    from demo.local_llm import generate,MODEL
    cases_path=ROOT/'experiments/fixtures/rag_cases.json'
    cases=json.loads(cases_path.read_text(encoding='utf-8'))
    corpus_hash=__import__('hashlib').sha256(json.dumps({d:corpus(d) for d in ['diabetes','stock','gold']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    identity=[sha(cases_path),corpus_hash,sha(Path(__file__)),sha(ROOT/'demo/domain_knowledge.py'),sha(ROOT/'demo/graph_retrieval.py'),sha(ROOT/'demo/local_llm.py'),sha(MODEL),generation]
    signature=__import__('hashlib').sha256(json.dumps(identity).encode()).hexdigest()[:12]
    folder=ROOT/'runs/rag'/signature
    if (folder/'complete.json').exists():
        check_run(folder);print('Verified cached RAG run',signature);return folder
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'config.json',dict(case_file='experiments/fixtures/rag_cases.json',case_sha256=sha(cases_path),
        corpus_sha256=corpus_hash,parameters=dict(rrf_k=60,k=3,generator_temperature=0,seed=42),
        method_chosen_before_test='hybrid',generation=generation,
        limits='Authored cases; not an independent population benchmark. Financial queries are keyed CSV lookups. Fact checks measure fixed expected facts, not general semantic faithfulness.'))
    records=[]
    for split in ['dev','test']:
        for case in cases[split]:
            for method in ['tfidf','bm25','kg','hybrid']:
                start=time.perf_counter();value=retrieve_domain(case['domain'],case['question'],method);ids=[d['id'] for d in value['sources']]
                rank=min([ids.index(g)+1 for g in case['gold_ids'] if g in ids],default=None)
                records.append(dict(split=split,case_id=case['id'],category=case['category'],method=method,question=case['question'],gold_ids=case['gold_ids'],retrieved=ids,rank=rank,
                    no_evidence_expected=not case['gold_ids'],abstained=value['status'] in ['no_evidence','out_of_scope'],latency_ms=(time.perf_counter()-start)*1000))
    write_json(folder/'retrieval_predictions.json',records)
    summary=[]
    for split in ['dev','test']:
        for category in ['conceptual','lookup','out_of_scope']:
            for method in ['tfidf','bm25','kg','hybrid']:
                rows=[r for r in records if r['split']==split and r['category']==category and r['method']==method]
                if not rows:continue
                summary.append(dict(split=split,category=category,method=method,n=len(rows),
                    hit_at_3=None if category=='out_of_scope' else float(np.mean([r['rank'] is not None for r in rows])),
                    mrr_at_3=None if category=='out_of_scope' else float(np.mean([1/r['rank'] if r['rank'] else 0 for r in rows])),
                    abstention_accuracy=float(np.mean([r['abstained'] for r in rows])) if category=='out_of_scope' else None))
    write_json(folder/'retrieval_metrics.json',summary)
    generated=[]
    partial=folder/'generation_partial.json'
    if partial.exists():generated=json.loads(partial.read_text(encoding='utf-8'))
    if generation:
        for case in cases['test']:
            if any(r['case_id']==case['id'] for r in generated):continue
            evidence=retrieve_domain(case['domain'],case['question'],'hybrid')
            if not evidence['sources']:
                generated.append(dict(case_id=case['id'],category=case['category'],question=case['question'],mode='abstain',answer=evidence['answer'],expected_abstention=not case['gold_ids']))
                write_json(partial,generated);continue
            print('Generate',case['id'],flush=True)
            value=generate(case['question'],evidence['sources']);text=normalize(value['answer'])
            facts=case.get('fact_groups',[])
            fact_pass=all(any(normalize(term) in text for term in alternatives) for alternatives in facts)
            # Numeric lookup tolerates display rounding, not another record's value.
            if 'number' in case:
                import re
                numbers=[float(n) for n in re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)?',value['answer'].replace(',','.'))]
                fact_pass=any(abs(n-case['number'])<=.011 for n in numbers)
            generated.append(dict(case_id=case['id'],category=case['category'],question=case['question'],mode='rag',
                answer=value['answer'],gold_answer=case.get('gold_answer'),citations=value['citations'],
                citation_ids_valid=set(value['citations'])<=set(d['id'] for d in evidence['sources']),
                cites_gold_source=bool(set(value['citations'])&set(case['gold_ids'])),expected_fact_pass=fact_pass,
                token_f1=token_f1(value['answer'],case.get('gold_answer','')),rouge_l=rouge_l(value['answer'],case.get('gold_answer','')),
                generation_ms=value['generation_ms']))
            write_json(partial,generated)
    write_json(folder/'generation_predictions.json',generated)
    generation_metrics=[]
    for category in ['conceptual','lookup']:
        rows=[r for r in generated if r['category']==category and r['mode']=='rag']
        if rows:generation_metrics.append(dict(category=category,n=len(rows),expected_fact_pass=float(np.mean([r['expected_fact_pass'] for r in rows])),
            valid_citation_rate=float(np.mean([r['citation_ids_valid'] for r in rows])),gold_citation_rate=float(np.mean([r['cites_gold_source'] for r in rows])),
            token_f1=float(np.mean([r['token_f1'] for r in rows])),rouge_l=float(np.mean([r['rouge_l'] for r in rows])),
            latency_p50_ms=float(np.median([r['generation_ms'] for r in rows]))))
    write_json(folder/'generation_metrics.json',generation_metrics)
    finish_run(folder,dict(generation_executed=generation,dev_cases=len(cases['dev']),test_cases=len(cases['test']),
        evaluation_completed=True,all_answers_semantically_proven=False))
    write_json(ROOT/'assets/data/rag_verified.json',dict(run=folder.relative_to(ROOT).as_posix(),retrieval=summary,generation=generation_metrics,
        note='Bộ câu tự soạn DEV/TEST; lookup và hỏi khái niệm được tách riêng. Kiểm tra fact cố định không thay thế đánh giá đầy đủ nội dung.'))
    print('RAG evaluation complete',folder,flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--without-generation',action='store_true');args=parser.parse_args();run(not args.without_generation)
