"""Explicit bounded graph walks and an inspectable BM25 lexical baseline."""
from collections import Counter,deque
import math,re
import numpy as np


def bm25(question,docs):
    from .knowledge import normalize
    stop={'la','gi','co','cua','cho','toi','nao','va','trong','bao','nhieu','ngay','the','ve','mot'}
    def tokens(text):return [w for w in re.findall(r'\w+',normalize(text)) if w not in stop]
    corpus=[tokens(d['title']+' '+d['text']) for d in docs];query=Counter(tokens(question))
    lengths=np.asarray([len(d) for d in corpus]);mean=float(lengths.mean()) or 1.;scores=np.zeros(len(docs))
    for word in query:
        df=sum(word in words for words in corpus);idf=math.log(1+(len(docs)-df+.5)/(df+.5))
        for i,words in enumerate(corpus):
            tf=words.count(word)
            scores[i]+=idf*(tf*2.5)/(tf+1.5*(.25+.75*lengths[i]/mean))
    return scores


def graph_paths(domain,question,docs):
    from .knowledge import normalize,NODES,EDGES
    q=normalize(question);paths={}
    if domain!='diabetes':
        for i,d in enumerate(docs):
            if d.get('date','not-a-date') in q:
                paths[i]=dict(score=3.,paths=[dict(nodes=['entity:'+d['date'],d['id'],d['id']+':'+field],
                    relations=['MENTIONS (reverse)','HAS_VALUE'],source=d['id']) for field in d.get('fields',{})])
        return paths
    aliases={'hba1c':['hba1c','a1c'],'type1':['type1'],'type2':['type2'],'symptoms':['trieu chung'],
             'thirst':['khat nhieu'],'urination':['tieu nhieu'],'prevention':['phong ngua'],
             'activity':['van dong','the duc'],'complications':['bien chung'],'tests':['xet nghiem','chan doan'],
             'brfss':['brfss','nhom tuoi'],'model':['mo hinh','mlp'],'insulin':['insulin'],'diabetes':['tieu duong']}
    seeds=[node for node,terms in aliases.items() if any(term in q for term in terms)]
    if len(seeds)>1 and 'diabetes' in seeds:seeds.remove('diabetes')
    adjacency={node:[] for node,_,_ in NODES}
    for a,b,relation,source in EDGES:
        adjacency.setdefault(a,[]).append((b,relation,source))
        adjacency.setdefault(b,[]).append((a,relation+' (reverse)',source))
    ids={d['id']:i for i,d in enumerate(docs)}
    for seed in seeds:
        weight=3. if seed in ['hba1c','type1','type2','brfss'] else .4 if seed=='insulin' else 1.
        queue=deque([(seed,[seed],[])]);seen={seed}
        while queue:
            node,nodes,relations=queue.popleft()
            if len(relations)>=2:continue
            for nxt,relation,source in adjacency.get(node,[]):
                if nxt in nodes:continue
                if source in ids:
                    index=ids[source];item=paths.setdefault(index,dict(score=0.,paths=[]))
                    item['score']+=weight/(len(relations)+1)
                    item['paths'].append(dict(nodes=nodes+[nxt],relations=relations+[relation],source=source))
                if nxt not in seen:
                    seen.add(nxt);queue.append((nxt,nodes+[nxt],relations+[relation]))
    return paths
