import itertools

from collections.abc import Callable, Sequence
from scipy.stats import hmean
from typing import Any

from .traces import StandardTrace, SummaryTrace, Summary, KeyTrace, BatchSummary
from .scorers import score_generic, score_str
from ..utils import is_data_model
from ..schemas import schema as schema_getter
from ...collections import Stream, ZippedStream

Scorer = Callable[[Any, Any], float|bool]


class Comparator:
    
    def __init__(self):
        self._primitive_scorers = dict[type, Scorer]([
            (str, score_str),
        ])
    
    def compare_batch[T, R](self, targets:dict[T, Any], refs:dict[R, Any]):
        assert len(targets) == len(refs)
        
        metrics = (
            ZippedStream((t, targets[t], r, refs[r]) for t, r in zip(targets, refs))
            .select(lambda t, td, r, rd: (t, r, self.compare(td, rd)))
            .zip_unwind(lambda t, r, s: ((t, r, field, trace) for field, trace in s.traces.items()))
            .groupby(lambda _, __, f, ___: f, lambda t, r, _, tr: KeyTrace(t, r, tr))
        )
        
        return BatchSummary(metrics)
    
    def compare(self, target:Any, ref:Any):
        assert is_data_model(target)
        assert is_data_model(ref)
        assert type(target) is type(ref)
        
        schema = schema_getter(target)
        target_realization = schema.realize(target, expand_none=True)
        target_endpoints = [*target_realization.endpoints]
        num_targets = len(set(
            Stream(target_endpoints)
            .select(schema.real_to_rep)
            .filter(lambda r: schema.rep.out_degree(r) == 0)
        ))
        
        ref_realization = schema.realize(ref, expand_none=True)
        ref_endpoints = [*ref_realization.endpoints]
        num_refs = len(set(
            Stream(ref_endpoints)
            .select(schema.real_to_rep)
            .filter(lambda r: schema.rep.out_degree(r) == 0)
        ))
        shared_fields = set(target_endpoints).union(ref_endpoints)
        
        traces = dict(
            Stream(shared_fields)
            .filter(lambda field: schema.rep.out_degree(schema.real_to_rep(field)) == 0)
            .split(lambda n: 
                (n, target_realization.value(n), ref_realization.value(n))
            )
            .select(lambda n, t, r: (n, self._compute_score(t, r)))
            .groupby(lambda n,_: schema.real_to_rep(n), lambda _,t:t)
            .map_key(lambda n: '.'.join(str(a) for a in n))
            .select(lambda _, traces:self._combine_trace(traces))
        )
        return Summary(num_targets, num_refs, traces)
    
    def _compute_score(self, target, ref):
        if type(target) is not type(ref):
            return StandardTrace(target, ref, 0)

        if isinstance(target, list|tuple):
            vals = [*itertools.zip_longest(target, ref)]
            item_traces = (
                ZippedStream(vals)
                .project(lambda t, r: self._compute_score(t, r))
                .evaluate()
            )
            if len(target) > 0:
                precision = hmean([max(t.score, 0.1) for t in item_traces[:len(target)]])
            else:
                precision = float(len(ref) == 0)

            score = precision
            if len(ref) > 0:
                recall = sum([t.score for t in item_traces[:len(ref)]], 0) / len(ref)
                score = hmean([max(precision, 0.1), recall])

            return SummaryTrace(
                score,
                item_traces
            )
        
        scorer = self._primitive_scorers.get(type(target), score_generic)
        return StandardTrace(
            target, ref,
            float(scorer(target, ref))
        ) 

    def _combine_trace(self, traces:Sequence[StandardTrace|SummaryTrace]):
        if len(traces) == 1:
            return traces[0]
        else:
            return SummaryTrace(
                hmean([max(t.score, 0.1) for t in traces]),
                traces
            )

    def customize[T](self, t:type[T], scorer:Callable[[T, T], bool|float]):
        self._primitive_scorers[t] = scorer
        return self
