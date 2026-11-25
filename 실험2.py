#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""소수파 게임: 무작위 참가자 모방 vs 최고점 모방 + 확률적 변이.

IDLE에서 F5 또는: python minority_chain_analysis.py
실행 즉시 로컬 웹사이트가 열린다. 웹 화면에서 시작/정지/속도/반복을 조절한다.
설치 없이 실행 가능. NumPy가 있으면 자동으로 사용한다.
빠른 확인: python minority_chain_analysis.py --quick
표준 라이브러리만 사용: python minority_chain_analysis.py --engine python
웹 조작 없이 전체 계산: python minority_chain_analysis.py --batch
결과: 새 결과 폴더의 chain_summary.html과 index.html, 회차별 비교 HTML, 원자료 CSV, 설정 JSON.
인터넷 없이 HTML을 열 수 있다. 모든 설정/정의는 보고서에도 기록된다.
"""
from bisect import bisect_right
from dataclasses import dataclass, asdict, replace
from copy import deepcopy
from heapq import merge
from itertools import groupby, islice, repeat
from array import array
from collections import Counter, deque
import base64
import struct
from pathlib import Path
from random import Random
from statistics import median
import argparse
import io
import shutil
import csv
import html
import json
import math
import platform
import sys
import tempfile
import time
import webbrowser
import threading
import secrets
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs, unquote

# ============ 여기서 실험 설정을 바꿀 수 있어 ============
@dataclass
class Config:
    n: int = 501                  # 에이전트 수
    q: int = 3                    # 선택지 수
    rounds: int = 30000            # 반복당 라운드
    repeats: int = 10              # 방식별 독립 반복 수
    burn: int = 10000              # 초기 적응 기간 (원자료에는 보존)
    score_limit: float = -5.0      # 개인 누적 점수가 이 값 미만이면 교체
    mutation_prob: float = 0.10    # 최고점 전략 복제 후 변이할 확률. 0=순수 복제
    mutation_sd: float = 0.01      # 변이 시 두 확률 사이에 이동할 양의 표준편차
    bins: int = 10                 # 각 확률축의 구간 수. 양 방식에 같은 구간 적용
    h_window: int = 50             # 직전 엔트로피 평균 구간
    h_drop: float = 0.02           # 정규화 엔트로피의 감소 기준
    loss_base: int = 50            # 집단 패배 후보 직전 비교 구간
    loss_block: int = 5            # 평균 기울기 블록 길이
    loss_repeats: int = 2          # 연속 저조 블록 수
    loss_sigma: float = 0.5        # 비교 평균 - 이 값 * 표준편차
    max_recovery: int = 2          # 비교 평균 이상으로 연속 회복할 수 있는 횟수
    vol_window: int = 50           # 선택 인원 비율의 시간 변동 측정 구간
    vol_ratio: float = 1.25        # 직전 구간 대비 변동 크기 증가 배율
    vol_hold: int = 5              # 증가 기준이 연속 충족돼야 하는 라운드
    compare_window: int = 200      # t_L 앞/뒤의 동일 길이 비교 구간
    max_lag: int = 200             # 시차 범위 -200~200
    lag_step: int = 5              # 시차 간격
    chain_horizon: int = 1000      # 첫 H부터 첫 V까지 허용 시간. 결과 보고 전 고정
    seed: int = 20261121
    probe_round: int = 0           # 0은 초기 적응 및 판정 이력이 끝난 다음 라운드

DEFAULT = Config()
MODES = ('random', 'best')
LABELS = {'random': '무작위 참가자 모방', 'best': '최고점 모방 + 확률적 변이'}
ENTROPIES = ('strategy', 'lineage')
HLABELS = {'strategy': '확률 분포 엔트로피', 'lineage': '계보 엔트로피'}
VERSION = '3.4-intervention-pairs'


def validate(c):
    if c.probe_round < 0:
        raise ValueError('개입 시작 라운드는 0 이상이어야 해.')
    if c.seed < 0:
        raise ValueError('시드는 0 이상이어야 해.')
    if c.n < 3 or not 2 <= c.q <= c.n or c.repeats < 1:
        raise ValueError('n>=3, 2<=q<=n, repeats>=1이어야 해.')
    if not (c.score_limit < 0 and 0 <= c.mutation_prob <= 1 and c.mutation_sd >= 0):
        raise ValueError('점수 기준은 음수, 변이 확률은 0~1, 변이 표준편차는 0 이상이어야 해.')
    if min(c.bins, c.h_window, c.loss_base, c.loss_block, c.loss_repeats,
           c.vol_window, c.vol_hold, c.compare_window, c.max_lag, c.lag_step, c.chain_horizon) < 1:
        raise ValueError('구간 길이와 시차 설정은 양의 정수여야 해.')
    if c.rounds < 1 or not 0 <= c.burn < c.rounds:
        raise ValueError('라운드는 1 이상, 초기 적응 기간은 0 이상 전체 라운드 미만이어야 해.')
    if not math.isfinite(c.mutation_sd):
        raise ValueError('변이 표준편차는 유한해야 해.')
    if not (0 < c.h_drop <= 1 and c.vol_ratio > 1 and c.loss_sigma >= 0 and c.max_recovery >= 0):
        raise ValueError('엔트로피 감소 기준/변동 배율/패배 판정 설정을 확인해 줘.')


def entropy(labels, n):
    """H=-sum(f log f)/log(N). 두 모드에 같은 분모 적용. 0~1."""
    return -sum((v/n)*math.log(v/n) for v in Counter(labels).values()) / math.log(n)


def fresh_strategy(rng, q):
    x = [rng.expovariate(1.0) for _ in range(q)]
    total = sum(x)
    return [a/total for a in x]  # Dirichlet(1,...,1)


def mutate(parent, rng, c):
    p = list(parent)
    if rng.random() >= c.mutation_prob or c.mutation_sd == 0:
        return p, False
    a, b = rng.sample(range(c.q), 2)
    s = p[a] + p[b]
    if s > 0:
        x = (p[a] + rng.gauss(0, c.mutation_sd)) % (2*s)
        p[a] = min(x, 2*s-x)
        p[b] = s-p[a]
    return p, True



LARGE_ROUNDS = 200_000
TILE_ROWS = 4096
TRACE_KEYS = ('h_strategy', 'h_lineage', 'drop_strategy', 'drop_lineage', 'gain', 'total',
              'slope', 'volatility', 'crowding', 'replacements', 'mutations', 'loss_degree', 'amp_pct')


class TailColumn:
    """최근 측정 창을 유지하면서 원래의 절대 라운드 인덱스를 지원한다."""
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, value):
        self.buffer[self.size % len(self.buffer)] = value
        self.size += 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self.size))]
        i = key if key >= 0 else self.size + key
        if not max(0, self.size-len(self.buffer)) <= i < self.size:
            raise IndexError(key)
        return self.buffer[i % len(self.buffer)]


class EventTimes:
    """한 종류의 모든 시작 시점. 확인 지연은 같은 판정 규칙에서 일정하다."""
    def __init__(self, delay):
        self.starts = array('Q')
        self.delay = delay

    def append(self, event):
        assert event['confirmed'] == event['t'] + self.delay
        self.starts.append(event['t'])

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        return {'t': t, 'confirmed': t + self.delay}

    def between(self, start, end):
        left = bisect_right(self.starts, start)
        right = max(left, bisect_right(self.starts, end-self.delay))
        return left, right


class RoundFile:
    """고정 길이 행을 순서대로 기록해 전체 길이와 무관한 크기의 창으로 읽는다."""
    def __init__(self, directory, q):
        self.record = struct.Struct('<' + 'd'*(len(TRACE_KEYS)+q))
        self.file = tempfile.NamedTemporaryFile(dir=directory, prefix='rounds_', suffix='.tmp',
                                                mode='wb', delete=False)
        self.path = Path(self.file.name)
        self.size = 0

    def append(self, trace):
        row = [trace[k][-1] for k in TRACE_KEYS]
        self.file.write(self.record.pack(*(math.nan if v is None else v for v in row), *trace['counts'][-1]))
        self.size += 1

    def flush(self):
        self.file.flush()

    def blocks(self, start, end, step=32768):
        with self.path.open('rb') as f:
            f.seek(start*self.record.size)
            for i in range(start, end, step):
                count = min(step, end-i)
                block = f.read(count*self.record.size)
                if len(block) != count*self.record.size:
                    raise IOError('라운드 기록을 끝까지 읽을 수 없어.')
                yield i, block

    def samples(self, lo, hi, stride):
        # 1부터 세는 전역 격자. 웹·저장 HTML 모두 같은 라운드를 표시한다.
        first = -(-(lo-1)//stride)*stride
        indices = range(first, hi, stride)
        if not indices:
            return b''
        with self.path.open('rb') as f:
            if stride == 1:
                f.seek(first*self.record.size)
                return f.read(len(indices)*self.record.size)
            result = bytearray()
            for index in indices:
                f.seek(index*self.record.size)
                result.extend(f.read(self.record.size))
            return bytes(result)

    def close(self):
        f = getattr(self, 'file', None)
        if f is not None:
            try:
                f.close()
                self.path.unlink(missing_ok=True)
            except OSError:
                pass

    def __del__(self):
        self.close()


class DiskColumn:
    def __init__(self, table, key):
        self.table = table
        self.index = None if key == 'counts' else TRACE_KEYS.index(key)

    def __len__(self):
        return self.table.size

    def values(self, start, end, np=None):
        if self.index is None:
            raise TypeError('counts는 행 단위로 읽어야 해.')
        chunks = self.table.blocks(start, end)
        if np is not None:
            length = self.table.record.size // 8
            parts = [np.frombuffer(raw, dtype='<f8').reshape(-1, length)[:, self.index]
                     for _, raw in chunks]
            return np.concatenate(parts) if parts else np.empty(0)
        pos = self.index
        return [row[pos] for _, raw in chunks for row in self.table.record.iter_unpack(raw)]


class FlagColumn:
    # 2는 아직 사건 여부를 확인할 수 없는 라운드다.
    def __init__(self, n):
        self.data = bytearray(b'\x02') * n

    def __len__(self):
        return len(self.data)

    def values(self, start, end, np=None):
        if np is None:
            return [math.nan if v == 2 else float(v) for v in self.data[start:end]]
        vals = np.frombuffer(self.data, dtype='u1', count=end-start, offset=start).astype(float)
        vals[vals == 2] = np.nan
        return vals


def disk_flags(trace, events, c, start, end, kind, np):
    n = end
    flags = {k: FlagColumn(n) for k in ('E','D','A')}
    span = c.loss_block*c.loss_repeats
    for key, lo, hi in (('E', max(start,c.h_window), end),
                        ('D', max(start,c.loss_base), end-span+1)):
        if hi > lo:
            flags[key].data[lo:hi] = bytes(hi-lo)
    first = max(start, 2*c.vol_window-1)
    stop = end-c.vol_hold+1
    for lo in range(first, stop, 32768):
        hi = min(stop,lo+32768)
        vals = trace['amp_pct'].values(lo,hi+c.vol_hold-1,np)
        if np is not None:
            prefix = np.concatenate(([0],np.cumsum(~np.isfinite(vals), dtype=np.int64)))
            flags['A'].data[lo:hi] = (2*(prefix[c.vol_hold:]!=prefix[:-c.vol_hold])).astype('u1').tobytes()
        else:
            streak = 0
            for i,v in enumerate(vals):
                streak = streak+1 if math.isfinite(v) else 0
                if streak >= c.vol_hold:
                    flags['A'].data[lo+i-c.vol_hold+1] = 0
    for dest,key in (('E',kind),('D','L'),('A','V')):
        group = events[key]
        left,right = group.between(start,end)
        for index in range(left,right):
            t = group.starts[index]-1
            if flags[dest].data[t] == 0:
                flags[dest].data[t] = 1
    return flags


def disk_lag_curve(x, y, c, start, end, np):
    # 구간 경계 너머의 시차 쌍도 포함해 두 구간의 중심화 합을 병합한다.
    lags = sorted({0,-c.max_lag,c.max_lag} | set(range(c.lag_step,c.max_lag+1,c.lag_step)) |
                  {-k for k in range(c.lag_step,c.max_lag+1,c.lag_step)})
    moments = {k: [0,0.,0.,0.,0.,0.] for k in lags}
    for lo in range(start,end,32768):
        hi = min(end,lo+32768)
        left_edge, right_edge = max(start,lo-c.max_lag), min(end,hi+c.max_lag)
        xs, ys = x.values(lo,hi,np), y.values(left_edge,right_edge,np)
        for k in lags:
            a,b = max(lo,start-k), min(hi,end-k)
            if a >= b:
                continue
            u,v = xs[a-lo:b-lo], ys[a+k-left_edge:b+k-left_edge]
            if np is not None:
                ok = np.isfinite(u) & np.isfinite(v)
                u,v = u[ok],v[ok]
                size = len(u)
                if not size:
                    continue
                mx,my = float(u.mean()),float(v.mean())
                du,dv = u-mx,v-my
                xx,yy,xy = float(np.sum(du*du)),float(np.sum(dv*dv)),float(np.sum(du*dv))
            else:
                pairs = [(p,q) for p,q in zip(u,v) if math.isfinite(p) and math.isfinite(q)]
                size = len(pairs)
                if not size:
                    continue
                mx = sum(p for p,_ in pairs)/size
                my = sum(q for _,q in pairs)/size
                xx = sum((p-mx)**2 for p,_ in pairs)
                yy = sum((q-my)**2 for _,q in pairs)
                xy = sum((p-mx)*(q-my) for p,q in pairs)
            count, mean_x, mean_y, sum_x, sum_y, cross = moments[k]
            total = count+size
            dx,dy = mx-mean_x,my-mean_y
            weight = count*size/total
            moments[k] = [total, mean_x+dx*size/total, mean_y+dy*size/total,
                          sum_x+xx+dx*dx*weight, sum_y+yy+dy*dy*weight, cross+xy+dx*dy*weight]
    rows = []
    for k,(count,_,__,xx,yy,xy) in moments.items():
        r = max(-1.,min(1.,xy/math.sqrt(xx*yy))) if count >= 3 and min(xx,yy)>1e-24 else None
        rows.append({'lag':k,'r':r,'n':count})
    return rows


def disk_phase_curves(trace, c, phase, kind, use_numpy, common, events, n=None):
    np = None
    if use_numpy:
        import numpy as np
    start,end = phase_range(c,phase,len(trace['loss_degree']) if n is None else n)
    data = ({'E':trace['drop_'+kind], 'D':trace['loss_degree'], 'A':trace['amp_pct']}
            if events is None else disk_flags(trace,events,c,start,end,kind,np))
    return {key: common if key == 'L_V' and common is not None else
            disk_lag_curve(data[x],data[y],c,start,end,np) for key,x,y in PAIRS}


def disk_windows(table, events, c):
    # 같은 W라운드 전후 정의를 적용하고 모든 L의 두 값을 시작 순서대로 기록한다.
    starts, w = events.starts, c.compare_window
    before = array('d',[math.nan]) * len(starts)
    after = array('d',[math.nan]) * len(starts)
    sums,squares = [0]*c.q,[0]*c.q
    tail = deque()
    ib,ia = bisect_right(starts,w),bisect_right(starts,0)
    for start, raw in table.blocks(0,table.size):
        for r,row in enumerate(table.record.iter_unpack(raw), start+1):
            counts = row[len(TRACE_KEYS):]
            old = tail.popleft() if len(tail)==w else (0,)*c.q
            tail.append(counts)
            for j,(now,prev) in enumerate(zip(counts,old)):
                now,prev = int(now),int(prev)
                sums[j] += now-prev
                squares[j] += now*now-prev*prev
            pre = ib<len(starts) and r==starts[ib]-1
            post = ia<len(starts) and r==starts[ia]+w-1
            if (pre or post) and r>=w:
                value = math.sqrt(sum(w*b-a*a for a,b in zip(sums,squares))/c.q)/(w*c.n)
                if pre:
                    before[ib] = value
                    ib += 1
                if post:
                    after[ia] = value
                    ia += 1
    for t,p,q in zip(starts,before,after):
        yield t, None if math.isnan(p) else p, None if math.isnan(q) else q


def write_large_trace(stem, table, c):
    # 아카이브의 CSV에는 모든 라운드, 그래프 타일에는 같은 행의 이진 표현을 저장한다.
    dest = stem.with_name(stem.name+'_tiles')
    dest.mkdir(exist_ok=True)
    levels = [1]
    while (c.rounds+levels[-1]-1)//levels[-1] > TILE_ROWS:
        levels.append(levels[-1]*8)
    buffers = {step:bytearray() for step in levels}
    pages = dict.fromkeys(levels,0)
    def flush(step):
        payload = base64.b64encode(buffers[step]).decode('ascii')
        name = f'{step}_{pages[step]}.js'
        (dest/name).write_text('acceptRoundTile(document.currentScript.src,"'+payload+'");',encoding='ascii')
        pages[step] += 1
        buffers[step].clear()
    fields = ['round','phase','H_strategy','H_lineage','entropy_drop_strategy','entropy_drop_lineage',
              'gain','cumulative_gain','mean_slope','temporal_rms','crowding_rms','replacements',
              'mutations','loss_per_agent','amplification_percent']+[f'count_{j+1}' for j in range(c.q)]
    with stem.with_name(stem.name+'_trace.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(fields)
        width=table.record.size
        for first,raw in table.blocks(0,table.size,TILE_ROWS):
            for i,row in enumerate(table.record.iter_unpack(raw),first):
                values=[None if math.isnan(x) else (int(x) if j in (9,10) or j>=len(TRACE_KEYS) else x)
                        for j,x in enumerate(row)]
                writer.writerow([i+1,'burn_in' if i<c.burn else 'observation',*values])
            for step in levels:
                off=(-first)%step
                buf=buffers[step]
                if step==1:
                    buf.extend(raw)
                    if len(buf)==TILE_ROWS*width:
                        flush(step)
                    continue
                for pos in range(off,len(raw)//width,step):
                    buf.extend(raw[pos*width:(pos+1)*width])
                    if len(buf)==TILE_ROWS*width:
                        flush(step)
    for step in levels:
        if buffers[step]:
            flush(step)


def write_large_analysis(path, events, phases):
    # JSON과 CSV 형식은 보통 회차와 같다. 긴 사건 목록은 묶음으로 직렬화한다.
    with path.open('w',encoding='utf-8') as f:
        f.write('{"events":{')
        for k,(key,group) in enumerate(events.items()):
            if k:
                f.write(',')
            f.write(json.dumps(key)+':[')
            for i in range(0,len(group),4096):
                if i:
                    f.write(',')
                items=[group[j] for j in range(i,min(i+4096,len(group)))]
                f.write(wire_json(items)[1:-1])
            f.write(']')
        f.write('},"phases":')
        json.dump(phases,f,ensure_ascii=False,allow_nan=False)
        f.write('}')


def start_trace():
    return {k: [] for k in ('h_strategy', 'h_lineage', 'counts', 'gain', 'total',
                            'replacements', 'mutations')}


def rounds_python(c, mode, run, progress=True, capture=False, trace=None, *,
                  checkpoint=None, initial=None, intervention=None):
    """같은 반복의 초기 전략과 행동 난수를 공유하고, 전략을 동시에 교체한다."""
    if initial is None:
        init, *streams = [Random(f'{c.seed}/{run}/{k}') for k in range(6)]
        ps = [fresh_strategy(init, c.q) for _ in range(c.n)]
        scores, lineages, total = [0]*c.n, list(range(c.n)), 0.0
    else:
        ps,scores,lineages,total,streams = deepcopy(initial)
    action,tie,best_donor,mutation,random_donor = streams
    arm,at,pulse_end,members,option = intervention[:5] if intervention else ('',0,0,(),0)
    tape = intervention[5] if intervention and len(intervention)>5 else None
    capture_at = checkpoint.at if checkpoint and not checkpoint.reason else -1
    trace = start_trace() if trace is None else trace
    def hs():
        cells = [tuple(min(c.bins-1, int(p*c.bins)) for p in row) for row in ps]
        return entropy(cells, c.n), entropy(lineages, c.n)
    h_s, h_l = hs()
    first_round = len(trace['gain'])+1 if initial is not None else 1
    for t in range(first_round,c.rounds+1):
        if t==capture_at:
            checkpoint.capture(trace,ps,scores,lineages,total,streams)
        if arm=='homogeneous' and t==at:
            mean = [sum(ps[i][j] for i in members)/len(members) for j in range(c.q)]
            for i in members:
                ps[i] = mean[:]
            h_s,h_l = hs()
        counts = [0]*c.q
        actions = []
        for row in ps:
            u = action.random()
            s = 0.0
            chosen = c.q-1
            for j, p in enumerate(row[:-1]):
                s += p
                if u < s:
                    chosen = j
                    break
            actions.append(chosen)
            counts[chosen] += 1
        if arm=='crowd' and at<=t<=pulse_end:
            for i in members:
                counts[actions[i]] -= 1
                actions[i] = option
                counts[option] += 1
        shield = arm=='loss_shield' and at<=t<=pulse_end
        if shield:
            actions = tape[t-at][0]
            counts = [0]*c.q
            for chosen in actions:
                counts[chosen] += 1
        smallest = min(counts)
        winner = tie.choice([j for j in range(c.q) if counts[j] == smallest])
        if shield:
            winner = tape[t-at][1]
        elif arm=='crowd' and at<=t<=pulse_end and tape is not None:
            tape.append((actions.copy(),winner))
        gain = (c.q*counts[winner]-c.n)/(c.q-1)
        total += gain
        losing_points = 0 if shield else -1
        scores = [s+(c.q-1 if a == winner else losing_points) for s,a in zip(scores,actions)]
        ids = [i for i, s in enumerate(scores) if s < c.score_limit*(c.q-1) - 1e-12]
        mutations = 0
        show = capture(t) if callable(capture) else capture
        view = {'round': t, 'counts': counts[:], 'actions': actions[:], 'probabilities': [p[:] for p in ps],
                'lineages': lineages[:], 'scores': [s/(c.q-1) for s in scores],
                'winner': winner, 'parents': [], 'replaced': ids[:]} if show else None
        # 해당 라운드의 전략 엔트로피를 교체 전에 기록한다.
        trace['h_strategy'].append(h_s)
        trace['h_lineage'].append(h_l)
        trace['counts'].append(counts)
        trace['gain'].append(gain)
        trace['total'].append(total)
        if ids:
            if mode == 'random':
                parents = []
                for i in ids:
                    # 균등 추첨의 후보 번호에서 자기 자리만 건너뛴다.
                    j = random_donor.randrange(c.n-1)
                    parents.append(j if j < i else j+1)
                children = [ps[j].copy() for j in parents]
            else:
                top_score = max(scores)
                top = [i for i, s in enumerate(scores) if s == top_score]
                parents = []
                for i in ids:
                    eligible = [j for j in top if j != i]
                    if not eligible:
                        second = max(s for j, s in enumerate(scores) if j != i)
                        eligible = [j for j, s in enumerate(scores) if j != i and s == second]
                    parents.append(best_donor.choice(eligible))
                children_flags = [mutate(ps[j], mutation, c) for j in parents]
                children = [x[0] for x in children_flags]
                mutations = sum(x[1] for x in children_flags)
            groups = [lineages[j] for j in parents]
            if show:
                view['parents'] = list(zip(ids, parents))
            for i, child, group in zip(ids, children, groups):
                ps[i], lineages[i], scores[i] = child, group, 0.0
            h_s, h_l = hs()
        trace['replacements'].append(len(ids))
        trace['mutations'].append(mutations)
        yield trace, view
        if progress and t % 5000 == 0:
            print(f'  {LABELS[mode]} {run}/{c.repeats}: {t:,}/{c.rounds:,}', flush=True)
    return trace


def rounds_numpy(c, mode, run, progress=True, capture=False, trace=None, *,
                 checkpoint=None, initial=None, intervention=None):
    import numpy as np
    if initial is None:
        init, *streams = [np.random.default_rng(s) for s in np.random.SeedSequence([c.seed,run]).spawn(6)]
        ps = init.dirichlet(np.ones(c.q), size=c.n)
        scores = np.zeros(c.n)
        lineages = np.arange(c.n, dtype=np.int64)
        total = 0.0
    else:
        ps,scores,lineages,total,streams = deepcopy(initial)
    action,tie,best_donor,mutation,random_donor = streams
    arm,at,pulse_end,members,option = intervention[:5] if intervention else ('',0,0,(),0)
    tape = intervention[5] if intervention and len(intervention)>5 else None
    capture_at = checkpoint.at if checkpoint and not checkpoint.reason else -1
    trace = start_trace() if trace is None else trace
    # 작은 격자는 같은 사전순 칸 번호를 한 정수로 세면 행 정렬을 생략할 수 있다.
    weights = (np.array([c.bins**k for k in reversed(range(c.q))],dtype=np.int64)
               if c.q*math.log2(c.bins)<=15 else None)
    def hs():
        cells = np.minimum(c.bins-1, (ps*c.bins).astype(int))
        if weights is None:
            _, nums = np.unique(cells, axis=0, return_counts=True)
        else:
            hist = np.bincount(cells @ weights)
            nums = hist[hist>0]
        f = nums/c.n
        h_s = -float(np.sum(f*np.log(f)))/math.log(c.n)
        return h_s, entropy(lineages.tolist(), c.n)
    h_s, h_l = hs()
    cutoffs = np.cumsum(ps, axis=1)[:, :-1]
    first_round = len(trace['gain'])+1 if initial is not None else 1
    for t in range(first_round,c.rounds+1):
        if t==capture_at:
            checkpoint.capture(trace,ps,scores,lineages,total,streams)
        if arm=='homogeneous' and t==at:
            ps[members] = ps[members].mean(axis=0)
            cutoffs[members] = np.cumsum(ps[members],axis=1)[:,:-1]
            h_s,h_l = hs()
        actions = (action.random((c.n, 1)) >= cutoffs).sum(axis=1)
        if arm=='crowd' and at<=t<=pulse_end:
            actions[members] = option
        shield = arm=='loss_shield' and at<=t<=pulse_end
        if shield:
            actions = tape[t-at][0]
        counts = np.bincount(actions, minlength=c.q)
        winner = int(tie.choice(np.flatnonzero(counts == counts.min())))
        if shield:
            winner = tape[t-at][1]
        elif arm=='crowd' and at<=t<=pulse_end and tape is not None:
            tape.append((actions.copy(),winner))
        gain = (c.q*int(counts[winner])-c.n)/(c.q-1)
        total += gain
        scores += np.where(actions == winner,c.q-1,0 if shield else -1)
        ids = np.flatnonzero(scores < c.score_limit*(c.q-1)-1e-12)
        mutations = 0
        show = capture(t) if callable(capture) else capture
        view = {'round': t, 'counts': counts.tolist(), 'actions': actions.tolist(), 'probabilities': ps.tolist(),
                'lineages': lineages.tolist(), 'scores': (scores/(c.q-1)).tolist(),
                'winner': winner, 'parents': [], 'replaced': ids.tolist()} if show else None
        trace['h_strategy'].append(h_s)
        trace['h_lineage'].append(h_l)
        trace['counts'].append(counts.tolist())
        trace['gain'].append(gain)
        trace['total'].append(total)
        if len(ids):
            if mode == 'random':
                parents = random_donor.integers(c.n-1, size=len(ids))
                parents += parents >= ids
            else:
                top = np.flatnonzero(scores == scores.max())
                parents = best_donor.choice(top, size=len(ids))
                # 자기 자신을 복제하지 않도록, 최고점 동률에서 독립적으로 재선택.
                for k in np.flatnonzero(parents == ids):
                    eligible = top[top != ids[k]]
                    if not len(eligible):
                        eligible = np.flatnonzero(np.arange(c.n) != ids[k])
                        eligible = eligible[scores[eligible] == scores[eligible].max()]
                    parents[k] = best_donor.choice(eligible)
            if show:
                view['parents'] = list(zip(ids.tolist(), parents.tolist()))
            children = ps[parents].copy()
            groups = lineages[parents].copy()
            if mode == 'best':
                rows = np.flatnonzero(mutation.random(len(ids)) < c.mutation_prob)
                if c.mutation_sd > 0 and len(rows):
                    a = mutation.integers(c.q, size=len(rows))
                    b = mutation.integers(c.q-1, size=len(rows))
                    b += b >= a
                    pair = children[rows, a] + children[rows, b]
                    delta = mutation.normal(0, c.mutation_sd, size=len(rows))
                    folded = np.remainder(children[rows, a]+delta, 2*pair,
                                          where=pair > 0, out=np.zeros(len(rows)))
                    x = np.minimum(folded, 2*pair-folded)
                    children[rows, a], children[rows, b] = x, pair-x
                    mutations = len(rows)
            ps[ids], lineages[ids], scores[ids] = children, groups, 0
            cutoffs[ids] = np.cumsum(children, axis=1)[:, :-1]
            h_s, h_l = hs()
        trace['replacements'].append(int(len(ids)))
        trace['mutations'].append(int(mutations))
        yield trace, view
        if progress and t % 5000 == 0:
            print(f'  {LABELS[mode]} {run}/{c.repeats}: {t:,}/{c.rounds:,}', flush=True)
    return trace


def simulate_python(c, mode, run, progress=True):
    for trace, _ in rounds_python(c, mode, run, progress, False):
        pass
    return trace


def simulate_numpy(c, mode, run, progress=True):
    for trace, _ in rounds_numpy(c, mode, run, progress, False):
        pass
    return trace

def temporal_rms(counts, n):
    """시간 평균 주변의 변동. 일정한 편향만 있으면 0. 단위는 인원 비율."""
    if not counts:
        return None
    q, m = len(counts[0]), len(counts)
    means = [sum(row[j] for row in counts)/m/n for j in range(q)]
    return math.sqrt(sum((row[j]/n-means[j])**2 for row in counts for j in range(q))/(m*q))

def correlation(x, y):
    if len(x) < 3:
        return None
    mx, my = sum(x)/len(x), sum(y)/len(y)
    vx = sum((v-mx)**2 for v in x)
    vy = sum((v-my)**2 for v in y)
    if vx <= 1e-24 or vy <= 1e-24:
        return None
    return max(-1.0, min(1.0, sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(vx*vy)))

# 측정기는 두 화면과 일괄 보고서가 함께 사용한다. 라운드 번호는 1부터 시작한다.
PHASES = ('all', 'early', 'late')
PHASE_NAMES = {'all': '전체', 'early': '초기 적응기', 'late': '적응 이후'}
PAIRS = (('H_L', 'E', 'D'), ('L_V', 'D', 'A'), ('H_V', 'E', 'A'))
SERIES = ('h_strategy', 'h_lineage', 'drop_strategy', 'drop_lineage', 'loss_degree',
          'amp_pct', 'volatility', 'slope', 'total')
MODE_COLORS = {'random': '#1769bd', 'best': '#ba510e'}
MODE_NAMES = {'random': '랜덤', 'best': '베스트'}


def phase_range(c, phase, n):
    if phase == 'early':
        return 0, min(n, c.burn)
    if phase == 'late':
        return min(n, c.burn), n
    return 0, n


class Measurements:
    """매 라운드에 한 번만 호출한다. 확인에 필요한 관측값이 모인 사건만 공개한다."""
    def __init__(self, c):
        self.c = c
        self.events = {key: [] for key in ('strategy', 'lineage', 'L', 'V')}
        self.h_sum = dict.fromkeys(ENTROPIES, 0.0)
        self.h_active = dict.fromkeys(ENTROPIES, False)
        self.loss_active = False
        self.vol_streak = 0
        self.count_sum = [0] * c.q
        self.count_sq = [0] * c.q

    def update(self, tr):
        c = self.c
        r = len(tr['gain'])
        i = r - 1
        for kind in ENTROPIES:
            h = tr['h_' + kind]
            previous = self.h_sum[kind] / c.h_window if i >= c.h_window else None
            drop = None if previous is None else previous - h[i]
            tr.setdefault('drop_' + kind, []).append(drop)
            hit = drop is not None and drop >= c.h_drop
            if hit and not self.h_active[kind]:
                self.events[kind].append({'t': r, 'confirmed': r})
            self.h_active[kind] = hit
            self.h_sum[kind] += h[i]
            if i >= c.h_window:
                self.h_sum[kind] -= h[i-c.h_window]

        g = tr['gain']
        tr.setdefault('loss_degree', []).append(-g[i] / c.n)
        slope = sum(g[-c.loss_block:]) / c.loss_block if r >= c.loss_block else None
        tr.setdefault('slope', []).append(slope)
        span = c.loss_block * c.loss_repeats
        if r >= c.loss_base + span:
            start = r - span
            prior = g[start-c.loss_base:start]
            mu = sum(prior) / c.loss_base
            variance = max(0.0, sum(x*x for x in prior) / c.loss_base - mu*mu)
            threshold = mu - c.loss_sigma * math.sqrt(variance)
            hit = all(sum(g[j:j+c.loss_block]) / c.loss_block < threshold
                      for j in range(start, r, c.loss_block))
            recovery = 0
            if hit:
                for value in g[start:r]:
                    recovery = recovery+1 if value >= mu else 0
                    if recovery > c.max_recovery:
                        hit = False
                        break
            if hit and not self.loss_active:
                self.events['L'].append({'t': start+1, 'confirmed': r})
            self.loss_active = hit

        counts, w = tr['counts'], c.vol_window
        for j, count in enumerate(counts[i]):
            self.count_sum[j] += count
            self.count_sq[j] += count*count
            if i >= w:
                old = counts[i-w][j]
                self.count_sum[j] -= old
                self.count_sq[j] -= old*old
        volatility = None
        if r >= w:
            # 정수로 분산의 분자를 누적해 일정한 인원 비율의 RMS는 정확히 0이 된다.
            num = sum(w*ss-s*s for s, ss in zip(self.count_sum, self.count_sq))
            volatility = math.sqrt(num/c.q) / (w*c.n)
        tr.setdefault('volatility', []).append(volatility)
        tr.setdefault('crowding', []).append(math.sqrt(sum((x/c.n-1/c.q)**2 for x in counts[i])/c.q))
        old = tr['volatility'][i-w] if r >= 2*w else None
        amp = 100*(volatility/old-1) if old is not None and old > 0 else None
        tr.setdefault('amp_pct', []).append(amp)
        hit = old is not None and old > 0 and volatility > c.vol_ratio*old
        self.vol_streak = self.vol_streak+1 if hit else 0
        if self.vol_streak == c.vol_hold:
            self.events['V'].append({'t': r-c.vol_hold+1, 'confirmed': r})


def lag_curve(x, y, c, use_numpy=False, start=None, end=None):
    """r(k) = corr(X[t], Y[t+k]); 양수 k에서 X가 먼저다. 결측은 쌍으로 제외한다."""
    start = c.burn if start is None else start
    end = min(len(x), len(y)) if end is None else min(end, len(x), len(y))
    lags = {0, -c.max_lag, c.max_lag}
    lags.update(range(c.lag_step, c.max_lag+1, c.lag_step))
    lags.update(-k for k in list(lags) if k > 0)
    rows = []
    if use_numpy:
        import numpy as np
        xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    for k in sorted(lags):
        lo, hi = max(start, start-k), min(end, end-k)
        r, size = None, 0
        if hi > lo and use_numpy:
            a, b = xx[lo:hi], yy[lo+k:hi+k]
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            size = len(a)
            if size >= 3:
                a, b = a-a.mean(), b-b.mean()
                va, vb = float(np.sum(a*a)), float(np.sum(b*b))
                if min(va, vb) > 1e-24:
                    r = max(-1.0, min(1.0, float(np.sum(a*b))/math.sqrt(va*vb)))
        elif hi > lo:
            a, b = [], []
            for t in range(lo, hi):
                if x[t] is not None and y[t+k] is not None:
                    a.append(x[t])
                    b.append(y[t+k])
            size, r = len(a), correlation(a, b)
        rows.append({'lag': k, 'r': r, 'n': size})
    return rows


def event_flags(trace, events, c, phase, kind):
    """시작과 확인을 관측할 수 있는 라운드만 0/1로 코딩한다."""
    n = len(trace['loss_degree'])
    start, end = phase_range(c, phase, n)
    flags = {k: [None]*n for k in ('E','D','A')}
    span = c.loss_block*c.loss_repeats
    for i in range(max(start,c.h_window), end):
        flags['E'][i] = 0
    for i in range(max(start,c.loss_base), end-span+1):
        flags['D'][i] = 0
    missing = [0]
    for value in trace['amp_pct']:
        missing.append(missing[-1]+int(value is None))
    for i in range(max(start,2*c.vol_window-1), end-c.vol_hold+1):
        if missing[i+c.vol_hold] == missing[i]:
            flags['A'][i] = 0
    for dest, key in (('E',kind),('D','L'),('A','V')):
        for event in events[key]:
            i = event['t']-1
            if start <= i < end and event['confirmed'] <= end and flags[dest][i] is not None:
                flags[dest][i] = 1
    return flags


def phase_curves(trace, c, phase, kind, use_numpy, common=None, events=None):
    if isinstance(trace.get('loss_degree'), DiskColumn):
        return disk_phase_curves(trace, c, phase, kind, use_numpy, common, events)
    start, end = phase_range(c, phase, len(trace['loss_degree']))
    if events is None:
        data = {'E': trace['drop_'+kind], 'D': trace['loss_degree'], 'A': trace['amp_pct']}
    else:
        data = event_flags(trace, events, c, phase, kind)
    return {key: common if key == 'L_V' and common is not None
            else lag_curve(data[left], data[right], c, use_numpy, start, end)
            for key, left, right in PAIRS}


def phase_summary(events, c, start, end, kind):
    first, counts = {}, {}
    for name,key in (('H',kind),('L','L'),('V','V')):
        group = events[key]
        if isinstance(group,EventTimes):
            lo,hi = group.between(start,end)
            first[name],counts[name] = (group[lo] if lo<hi else None),hi-lo
        else:
            selected = [e for e in group if start<e['t'] and e['confirmed']<=end]
            first[name],counts[name] = (selected[0] if selected else None),len(selected)
    h, l, v = (first[k]['t'] if first[k] else None for k in ('H', 'L', 'V'))
    ordered = None if None in (h, l, v) else h < l < v and v-h <= c.chain_horizon
    return {'first': first, 'counts': counts,
            'HL': l-h if h is not None and l is not None else None,
            'LV': v-l if v is not None and l is not None else None, 'chain': ordered}


def window_changes(trace, events, c):
    # 모든 L에서 바로 전 W라운드와 L부터의 W라운드를 비교한다.
    counts, w = trace['counts'], c.compare_window
    queries = {r for event in events for r in (event['t']-1, event['t']-1+w) if w <= r <= len(counts)}
    if not queries:
        return {}
    sums, squares, values = [0]*c.q, [0]*c.q, {}
    for r, row in enumerate(counts, 1):
        for j, count in enumerate(row):
            sums[j] += count
            squares[j] += count*count
            if r > w:
                old = counts[r-1-w][j]
                sums[j] -= old
                squares[j] -= old*old
        if r in queries:
            values[r] = math.sqrt(sum(w*b-a*a for a, b in zip(sums, squares))/c.q)/(w*c.n)
    return {str(e['t']): [values.get(e['t']-1), values.get(e['t']-1+w)] for e in events}


class PairedRun:
    def __init__(self, c, run, use_numpy, capture=False, directory=None):
        generator = rounds_numpy if use_numpy else rounds_python
        self.tables = {}
        self.meters = {m: Measurements(c) for m in MODES}
        self.studies = {m:InterventionPair(c,m,run,self.meters[m]) for m in MODES}
        self.traces, self.iterators = {}, {}
        for m in MODES:
            tail = None
            if c.rounds >= LARGE_ROUNDS:
                cap = 2 + max(c.h_window,c.vol_window,c.loss_base+c.loss_block*c.loss_repeats)
                keys = (*TRACE_KEYS,'counts')
                tail = {key:TailColumn(cap) for key in keys}
                table = self.tables[m] = RoundFile(directory,c.q)
                self.traces[m] = {k:DiskColumn(table,k) for k in keys}
                delay = {'L':c.loss_block*c.loss_repeats-1, 'V':c.vol_hold-1}
                self.meters[m].events = {k:EventTimes(delay.get(k,0)) for k in (*ENTROPIES,'L','V')}
            else:
                self.traces[m] = start_trace()
            self.iterators[m] = generator(c,m,run,False,capture,tail,checkpoint=self.studies[m])
        self.views = dict.fromkeys(MODES)
        self.round = 0

    def advance(self):
        for m in MODES:
            trace, view = next(self.iterators[m])
            self.meters[m].update(trace)
            if self.tables:
                self.tables[m].append(trace)
            else:
                self.traces[m] = trace
            if view is not None:
                self.views[m] = view
        self.round += 1


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_trace(path, trace, c):
    fields = ['round', 'phase', 'H_strategy', 'H_lineage', 'entropy_drop_strategy',
              'entropy_drop_lineage', 'gain', 'cumulative_gain', 'mean_slope',
              'temporal_rms', 'crowding_rms', 'replacements', 'mutations', 'loss_per_agent',
              'amplification_percent'] + [f'count_{j+1}' for j in range(c.q)]
    keys = ('h_strategy', 'h_lineage', 'drop_strategy', 'drop_lineage', 'gain', 'total',
            'slope', 'volatility', 'crowding', 'replacements', 'mutations', 'loss_degree', 'amp_pct')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, counts in enumerate(trace['counts']):
            writer.writerow([i+1, 'burn_in' if i<c.burn else 'observation',
                             *(trace[k][i] for k in keys), *counts])

def analyze(trace, c, use_numpy=False, phase='late'):
    """기존 Python 호출을 위한 회차별 분석. phase='all'이면 초기 적응기도 포함한다."""
    if phase not in PHASES:
        raise ValueError('phase는 all, early, late 중 하나여야 해.')
    working = start_trace()
    meter = Measurements(c)
    raw_keys = ('gain', 'counts', 'h_strategy', 'h_lineage')
    for values in zip(*(trace[k] for k in raw_keys)):
        for k, value in zip(raw_keys, values):
            working[k].append(value)
        meter.update(working)
    for key in (*SERIES, 'crowding'):
        if key in working and key not in ('h_strategy','h_lineage','total'):
            trace[key] = working[key]
    for kind in ENTROPIES:
        trace['h_cutoff_'+kind] = [None if d is None else h+d-c.h_drop
                                   for h,d in zip(trace['h_'+kind],trace['drop_'+kind])]
    n = len(trace['gain'])
    start, end = phase_range(c, phase, n)
    selected = {k: [v for v in group if start<v['t'] and v['confirmed']<=end]
                for k, group in meter.events.items()}
    loss, vol = selected['L'], selected['V']
    tl = loss[0]['t'] if loss else None
    tv = vol[0]['t'] if vol else None
    windows = window_changes(trace, loss[:1], c)
    pre, post = windows.get(str(tl), [None, None])
    result = {'common': {'t_L': tl, 't_V': tv, 'loss_events': loss, 'vol_events': vol,
        'pre_rms': pre, 'post_rms': post, 'post_pre_ratio': post/pre if pre and post is not None else None,
        'window_complete': pre is not None and post is not None,
        'pre_includes_burn': tl is not None and tl-1-c.compare_window<c.burn,
        't_L_confirmed': loss[0]['confirmed'] if loss else None,
        't_V_confirmed': vol[0]['confirmed'] if vol else None}, 'entropy': {}}
    common = None
    for kind in ENTROPIES:
        summary = phase_summary(meter.events, c, start, end, kind)
        th = selected[kind][0]['t'] if selected[kind] else None
        found = None not in (th, tl, tv)
        curves = phase_curves(trace, c, phase, kind, use_numpy, common)
        common = curves['L_V']
        result['entropy'][kind] = {'t_H': th, 'events': selected[kind], 'all_found': found,
            'ordered': found and th<tl<tv, 'chain': bool(summary['chain']),
            'delta_L_minus_H': summary['HL'], 'lag': curves}
    return result

def methods(c):
    span = c.loss_block*c.loss_repeats
    return f"""<section class="definitions" id="definitions"><h2>세 사건의 판정 기준</h2>
<p>개인 점수가 {c.score_limit:g}점 미만이면 랜덤은 다른 {c.n-1:,}명 중 한 명을 같은 확률로 뽑아 그 사람의 선택 확률 벡터를 그대로 복사해. 베스트는 다른 참가자 중 최고점자의 전략을 복사하고 {c.mutation_prob:.1%} 확률로 변이를 줘. 두 방식 모두 그 라운드의 교체 직전 전략과 계보를 물려받아.</p>
<div class="definitions-grid"><article><span class="symbol">H↓</span><h3>엔트로피 감소</h3>
<p>라운드 시작 시 각 에이전트의 전략은 선택지별 확률 벡터다. 각 확률축을 {c.bins}등분해 같은 칸에 속한 에이전트 비율 f를 구한다. 전략이 여러 칸에 고르게 퍼질수록 H가 커진다.</p>
<code>H = −Σ f ln(f) / ln(N)<br>E(t) = 직전 {c.h_window}회 평균 H − H(t)</code>
<p>E가 {c.h_drop:g} 이상인 연속 구간의 첫 라운드가 t<sub>H</sub>다. 첫 판정 가능 시점은 {c.h_window+1}회다. 계보 보기에서는 공통 조상을 가진 집단별 비율로 H를 구한다.</p></article>
<article><span class="symbol">L</span><h3>집단 패배</h3>
<p>최소 인원 선택지가 승리한다. 동률이면 한 선택지를 균등 추첨한다. 개인 보상은 승리 +1점, 패배 −1/(q−1)점이다. N은 전체 인원, q는 선택지 수, w는 승리 인원이다.</p>
<code>G(t) = (q·w − N)/(q−1)<br>D(t) = −G(t)/N</code>
<p>후보 라운드 직전 {c.loss_base}회 D의 평균 μ와 모집단 표준편차 σ를 구한다. 이어지는 {c.loss_block}회씩 {c.loss_repeats}개 블록의 평균 D가 모두 μ + {c.loss_sigma:g}σ를 초과하고, D ≤ μ인 회복이 {c.max_recovery+1}회 연속 일어나지 않으면 첫 라운드를 t<sub>L</sub>로 기록한다. 첫 판정 가능 시작점은 {c.loss_base+1}회이며 시작 {span-1}라운드 뒤 확인된다.</p></article>
<article><span class="symbol">V↑</span><h3>진동 증폭</h3>
<p>선택지별 인원 비율이 최근 {c.vol_window}라운드 동안 각 선택지의 구간 평균에서 얼마나 움직였는지 잰다. 모든 선택지와 라운드에서 편차를 제곱해 평균한 뒤 제곱근을 취한 값이 V다.</p>
<code>A(t) = 100 × [V(t)/V(t−{c.vol_window}) − 1]</code>
<p>직전 구간보다 V가 {100*(c.vol_ratio-1):g}% 넘게 커진 상태가 {c.vol_hold}회 연속이면 그 첫 라운드가 t<sub>V</sub>다. 첫 판정 가능 시작점은 {2*c.vol_window}회다. 기준 구간 V가 0이면 증가율을 계산할 수 없다. 여기서 진동은 인원 비율의 시간 변동을 뜻하며 주기성은 별도 가설이다.</p></article></div>
<p class="note">세 사건은 서로 독립적으로 찾고 연속 충족을 한 사건으로 센다. 표의 순서 판정은 선택한 구간에서 처음 확인된 각 사건의 시작점이 t<sub>H</sub> &lt; t<sub>L</sub> &lt; t<sub>V</sub>이고, 처음과 끝의 간격이 {c.chain_horizon:,}라운드 이내일 때 성립한다. 시간 순서와 상관계수로 연관성을 살펴볼 수 있다. 별도 개입 실험에서는 같은 상태에서 전략 또는 선택을 바꿨을 때의 효과를 비교한다.</p></section>"""


def readme(c, engine, quick):
    return f"""소수파 게임 · 랜덤과 베스트 동시 비교 {VERSION}
실험 화면에서 속도를 고르거나 초당 라운드를 직접 입력한다. 설정 속도와 두 방식이 함께 마친 실측 속도를 확인할 수 있다. 회차당 200,000라운드부터는 전체 원자료를 디스크에 기록하고 그래프를 구간별로 읽는다. 긴 회차의 저장 HTML은 결과 폴더의 *_tiles 폴더와 함께 연다.
Python 파일을 IDLE에서 F5로 실행하면 브라우저가 열린다. ‘두 방식 시작’을 한 번 누르면 같은 회차를 같은 라운드까지 함께 계산하고 다음 반복을 자동 진행한다.
현재 설정은 {c.n}명, 선택지 {c.q}개, {c.rounds:,}라운드, {c.repeats}회, 시드 {c.seed}, 엔진 {engine}이다.
랜덤은 파랑, 베스트는 주황이다. 실험 중의 색은 모든 그래프와 저장 파일에서 같은 방식을 뜻한다.
그래프마다 같은 축 위에서 두 방식을 비교한다. 분석 구간의 기본값은 전체다. 초기 적응기는 첫 {c.burn:,}라운드다. 엔트로피는 직전 평균과 현재 값, 패배는 직전 손실 분포와 이어지는 연속 블록, 증폭은 겹치지 않는 두 구간의 변동 크기를 비교한다. 사건의 상세 판정 기준은 HTML의 ‘세 사건의 판정 기준’에 있다.
설정된 마지막 라운드를 마치면 연쇄관계 종합 결과가 같은 화면에 자동으로 표시된다. chain_summary.html은 단독으로 열 수 있는 회차별·누적 종합 결과이고, index.html은 반복 요약, compare_01.html 등은 두 방식의 회차별 독립형 비교 화면이다. random, best 폴더의 CSV에 전체 정밀도의 원자료, 모든 사건의 시작·확인, 구간별 상관계수와 짝 수, 패배 전후 변동 크기가 기록된다.
전략 교체는 개인 점수가 {c.score_limit:g}점 미만일 때 일어난다. 랜덤은 자신을 제외한 {c.n-1:,}명 중 한 명을 균등하게 뽑아 교체 직전의 확률 벡터와 계보를 그대로 복사한다. 베스트는 다른 참가자 중 최고점자의 전략과 계보를 복제하고 {c.mutation_prob:g}의 확률로 변이를 준다. 두 확률 사이 이동량의 표준편차는 {c.mutation_sd:g}다.
같은 회차의 두 방식에서 초기 전략과 행동 난수열을 맞춘다. 교체와 모방은 각 집단 안에서 진행된다. 같은 엔진·설정·시드로 재실행하면 재현된다.
빠른 실험은 --quick, 자동 계산은 --batch, NumPy 없이 실행하려면 --engine python을 사용한다.
마지막 회차가 끝나면 chain_evidence.html에 같은 라운드 대조와 개입군·대조군 짝비교가 표시된다. 원자료는 chain_evidence.json과 run_번호_interventions.json에 있다.
기존 결과는 웹 화면에서 ‘기존 결과 폴더 분석’을 열거나 --evidence-from 옵션에 config.json이 있는 폴더를 지정한다.
"""


def wire_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', r'\u003c')


def write_lag_csv(path, phases):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['phase','entropy','basis','pair','lag','r','n_pairs'])
        for phase, by_kind in phases.items():
            for kind, entry in by_kind.items():
                for basis in ('curves','event_curves'):
                    for key, curve in entry[basis].items():
                        w.writerows((phase,kind,'continuous' if basis=='curves' else 'events',key,
                                     v['lag'],v['r'],v['n']) for v in curve)


# 완료한 라운드만 사용한 연쇄 요약. 통계는 원자료에서 구하고 화면과 파일이 같은 값을 쓴다.
def follow_entropy_events(events, c):
    """각 H 뒤의 열린 구간 (H, H+horizon]에서 L 다음 V가 한 번이라도 있었는지 센다."""
    def starts(key):
        group = events[key]
        if isinstance(group,EventTimes):
            return group.starts
        return [e['t'] for e in group if e['confirmed'] <= c.rounds]
    h_starts,losses,rises = (starts(key) for key in ('strategy','L','V'))
    # 구간 맨 끝에 시작한 후보도 확인할 수 있어야 모든 H에 같은 추적 기회를 준다.
    confirm_delay = max(c.loss_block*c.loss_repeats-1, c.vol_hold-1)
    eligible = bisect_right(h_starts,c.rounds-c.chain_horizon-confirm_delay)
    followed = 0
    for idx in range(eligible):
        h = h_starts[idx]
        li = bisect_right(losses, h)
        if li == len(losses) or losses[li] >= h+c.chain_horizon:
            continue
        vi = bisect_right(rises, losses[li])
        if vi < len(rises) and rises[vi] <= h+c.chain_horizon:
            followed += 1
    return {'matched': followed, 'eligible': eligible, 'short': len(h_starts)-eligible}


def strongest_lag(curve):
    valid = (v for v in curve if v['r'] is not None)
    return max(valid, key=lambda v: (abs(v['r']), -abs(v['lag']), -v['lag']), default=None)


def completed_side(c, events, phases, ratios):
    report = phases['all']['strategy']
    valid = [r for r in ratios if r is not None]
    return {'firsts': report['summary'], 'late': phases['late']['strategy']['summary']['chain'],
            'follow': follow_entropy_events(events, c), 'windows': len(valid),
            'larger': sum(r > 1 for r in valid), 'ratio': median(valid) if valid else None,
            'peaks': {key: strongest_lag(curve) for key, curve in report['curves'].items()}}


def verdict(s, c):
    prefix = {'H': '엔트로피 감소', 'L': '집단 패배', 'V': '진동 증폭'}
    points = s['first']
    missing = [name for k, name in prefix.items() if points[k] is None]
    if missing:
        return '판정 보류', ' · '.join(missing)+' 미관측'
    h, l, v = (points[k]['t'] for k in ('H','L','V'))
    if s['chain']:
        return '연쇄 충족', f'H에서 L까지 {l-h:,}라운드, L에서 V까지 {v-l:,}라운드'
    if not h < l < v:
        return '순서 미충족', '세 첫 사건의 시작 라운드가 H → L → V 순서를 충족하지 않았어.'
    return '시간 한도 초과', f'H부터 V까지 {v-h:,}라운드로 기준 {c.chain_horizon:,}라운드를 초과했어.'


def result_table(rows, title):
    cells = ''.join(f'<tr><th scope="row">{label}</th><td>{a}</td><td>{b}</td></tr>' for label, a, b in rows)
    return (f'<div class="scroll"><table class="result-table" aria-label="{title}"><thead><tr>'
            '<th scope="col">지표</th><th scope="col" class="random">랜덤</th><th scope="col" class="best">베스트</th>'
            f'</tr></thead><tbody>{cells}</tbody></table></div>')


def first_card(mode, side, c):
    summary = side['firsts']
    label, reason = verdict(summary,c)
    steps = []
    for k, name in (('H','H↓ 엔트로피 감소'),('L','L 집단 패배'),('V','V↑ 진동 증폭')):
        event = summary['first'][k]
        value = f"{event['t']:,}회" if event else '미관측'
        steps.append(f'<div><dt>{name}</dt><dd>{value}</dd></div>')
    return (f'<article class="result-card {mode}"><p class="method-name">{MODE_NAMES[mode]} · {LABELS[mode]}</p>'
            f'<h3>{label}</h3><dl class="milestones">{"".join(steps)}</dl><p>{reason}</p></article>')


def count_text(matched, eligible, unit):
    if not eligible:
        return '평가 가능한 사례 없음'
    return f'{matched:,} / {eligible:,}{unit} ({100*matched/eligible:.1f}%)'


def comparison_evidence(c, groups, peaks, multi):
    # 표의 각 열은 한 방식이며, 반복 평균은 각 회차에 동등한 가중치를 준다.
    rows = []
    for mode in MODES:
        sides = groups[mode]
        counts = [sum(x['firsts']['counts'][k] for x in sides) for k in ('H','L','V')]
        follow = {k: sum(x['follow'][k] for x in sides) for k in ('matched','eligible','short')}
        valid = sum(x['windows'] for x in sides)
        larger = sum(x['larger'] for x in sides)
        ratios = [x['ratio'] for x in sides if x['ratio'] is not None]
        ratio = f'{median(ratios):.3f}배' if ratios else '계산 불가'
        fraction = count_text(follow['matched'],follow['eligible'],'건')
        if follow['short']:
            fraction += f"<small>추적 구간 부족 {follow['short']:,}건</small>"
        elif counts[0] == 0:
            fraction = 'H↓ 미관측'
        rows.append([' / '.join(f'{n:,}' for n in counts), fraction, count_text(larger,valid,'건'), ratio])
    labels = ['사건 수 H↓ / L / V↑', 'H↓ 뒤 L → V↑가 이어진 비율',
              '패배 후 변동이 더 커진 비율', '사후 / 사전 변동 크기의 중앙값']
    table = result_table(zip(labels,*rows), '전체 라운드의 연쇄 지표')
    lag_rows = []
    for key, label in (('H_L','엔트로피 감소량 → 손실'),('L_V','손실 → 진동 증폭률'),('H_V','엔트로피 감소량 → 진동 증폭률')):
        values = []
        for mode in MODES:
            peak = peaks[mode][key]
            if peak is None:
                values.append('계산 불가')
            else:
                sample = f"{peak['n_runs']:,}회차" if multi else f"{peak['n']:,}쌍"
                values.append(f"r = {peak['r']:+.3f}<small>시차 {peak['lag']:+,}라운드 · {sample}</small>")
        lag_rows.append((label,*values))
    lag_title = '회차별 상관을 평균한 뒤 |r|가 가장 큰 시차' if multi else '관측 구간에서 |r|가 가장 큰 시차'
    median_help = ' 여러 회차를 볼 때는 회차별 중앙값의 중앙값을 표시해.' if multi else ''
    return (table+'<details class="result-evidence"><summary>집계 기준과 시차 상관</summary>'
            f'<p>H↓ 하나마다 다음 {c.chain_horizon:,}라운드 안에 L, 그 뒤 V↑가 있었는지 한 번씩 세어. '
            '마지막 후보를 확인할 기간까지 확보한 H↓가 분모야. 가까운 H↓들이 같은 L이나 V↑를 공유할 수 있어.</p>'
            f'<p>패배 전후 비율은 패배 직전 {c.compare_window:,}라운드와 시작부터 {c.compare_window:,}라운드의 변동 크기(V)를 비교해. '
            '두 구간이 완전하고 사전 V가 양수인 패배만 분모에 들어가.'+median_help+'</p>'
            f'<h4>{lag_title}</h4>'+result_table(lag_rows,lag_title)+
            '<p>양의 시차는 앞 변수가 먼저, 음의 시차는 뒤 변수가 먼저라는 뜻이야. 시간적 연쇄와 상관계수는 관측된 관계를 보여 줘.</p></details>')


def run_result(c, run, sides):
    cards = ''.join(first_card(m,sides[m],c) for m in MODES)
    groups = {m:[sides[m]] for m in MODES}
    peaks = {m:sides[m]['peaks'] for m in MODES}
    return (f'<h3 class="result-subtitle">{run}회차 · 1–{c.rounds:,}라운드</h3>'
            '<div class="result-cards">'+cards+'</div>'+comparison_evidence(c,groups,peaks,False))


def overall_result(c, per_run, aggregate):
    n = len(per_run)
    groups = {m:[r[m] for r in per_run.values()] for m in MODES}
    cards = []
    for m, sides in groups.items():
        states = [x['firsts']['chain'] for x in sides]
        yes, no, unknown = states.count(True), states.count(False), states.count(None)
        title = f'{yes} / {n}회 연쇄 충족' if yes+no else f'{unknown}회 모두 판정 보류'
        later = [x['late'] for x in sides]
        late_text = (f'<p class="later">적응 이후 첫 사건 · 충족 {later.count(True)}회 · 미충족 {later.count(False)}회 · 보류 {later.count(None)}회</p>'
                     if c.burn else '')
        cards.append(f'<article class="result-card {m}"><p class="method-name">{MODE_NAMES[m]} · {LABELS[m]}</p>'
                     f'<h3>{title}</h3><p>충족 {yes}회 · 미충족 {no}회 · 보류 {unknown}회</p>{late_text}</article>')
    peaks = {m:{k:strongest_lag(curve) for k,curve in aggregate['all']['strategy']['curves'][m].items()} for m in MODES}
    return (f'<h3 class="result-subtitle">완료된 {n}회차 종합 · 회차당 {c.rounds:,}라운드</h3>'
            '<div class="result-cards">'+''.join(cards)+'</div>'+comparison_evidence(c,groups,peaks,True))


def probe_schedule(c):
    delay_l = c.loss_block*c.loss_repeats - 1
    delay_v = c.vol_hold - 1
    history = max(c.h_window, c.loss_base, 2*c.vol_window)
    at = c.probe_round or max(c.burn, history)+1
    pulse_end = at+delay_l
    stop = max(at+c.chain_horizon+max(delay_l, delay_v), pulse_end+delay_l)
    if at <= history:
        return at, stop, '개입 전 판정 이력이 부족해.'
    if stop > c.rounds:
        return at, stop, '후속 관측과 사건 확인에 필요한 라운드가 부족해.'
    return at, stop, None


def probe_outcomes(c, at, groups, h, mean_loss):
    pulse_end = at+c.loss_block*c.loss_repeats-1
    end = at+c.chain_horizon
    clean = pulse_end+2*c.vol_window
    def select(key, high):
        g = groups[key]
        if isinstance(g,EventTimes):
            left = bisect_right(g.starts,at-1)
            right = bisect_right(g.starts,high)
            return g.starts[left:right]
        return [e['t'] for e in g if at<=e['t']<=high]
    vs = select('V',end)
    ls = select('L',max(pulse_end,end))
    h_event = bool(select('strategy',at))
    first = next((t for t in ls if at<t<=end),None)
    follow = next((v for v in vs if first is not None and first<v),None)
    peak = next((v for v in vs if clean<=v),None)
    return {'h':float(h),'pulse_loss':float(mean_loss),'h_event':h_event,
            'l_event':any(t<=pulse_end for t in ls),'HL':first is not None,
            'HLV':follow is not None,'LV':(peak is not None) if clean<=end else None,
            'wait_HL':(first if first is not None else end+1)-at,
            'wait_HLV':(follow if follow is not None else end+1)-at,
            'wait_LV':((peak if peak is not None else end+1)-clean+1) if clean<=end else None}


def recorded_probe(c,trace,events):
    at,_,reason = probe_schedule(c)
    if reason:
        return None
    def values(key,start,end):
        col = trace[key]
        return col.values(start,end) if isinstance(col,DiskColumn) else col[start:end]
    span = c.loss_block*c.loss_repeats
    return probe_outcomes(c,at,events,values('h_strategy',at-1,at)[0],
                          math.fsum(values('loss_degree',at-1,at-1+span))/span)

class InterventionPair:
    """실제 회차의 개입 직전 상태를 보관한다. 원래 게임은 자체 난수로 계속 진행한다."""
    def __init__(self, c, mode, run, meter):
        self.at, self.stop, self.reason = probe_schedule(c)
        self.c, self.mode, self.run, self.meter = c, mode, run, meter
        self.saved = None

    def capture(self, trace, ps, scores, lineages, total, streams):
        c = self.c
        keep = 2+max(c.h_window, c.vol_window, c.loss_base+c.loss_block*c.loss_repeats)
        history = {}
        for key, col in trace.items():
            recent = col[-keep:]
            tail = TailColumn(keep)
            tail.size = len(col)-len(recent)
            for item in recent:
                tail.append(item)
            history[key] = tail
        accum = {k:v for k,v in vars(self.meter).items() if k not in ('c','events')}
        self.saved = (deepcopy((ps,scores,lineages,total,streams)), history, deepcopy(accum))

    def calculate(self, use_numpy, expected):
        c = self.c
        if self.reason:
            return {'at': self.at, 'reason': self.reason}
        if self.saved is None:
            raise RuntimeError('개입 직전의 상태를 찾을 수 없어.')
        rng = Random(f'{c.seed}/{self.run}/intervention')
        members = sorted(rng.sample(range(c.n), max(2, c.n//2)))
        option = rng.randrange(c.q)
        gen = rounds_numpy if use_numpy else rounds_python
        sim, history, accum = self.saved
        span = c.loss_block*c.loss_repeats
        pulse_end = self.at+span-1
        horizon_end = self.at+c.chain_horizon
        first_clean = pulse_end+2*c.vol_window
        outcomes = {}
        short = replace(c, rounds=self.stop)
        pulse_tape = []
        for arm in ('control','homogeneous','crowd','loss_shield'):
            trace = deepcopy(history)
            meter = Measurements(short)
            vars(meter).update(deepcopy(accum))
            plan = (arm,self.at,pulse_end,members,option,pulse_tape)
            rounds = gen(short,self.mode,self.run,False,False,trace,
                         initial=sim,intervention=plan)
            pulse_d = []
            for t in range(self.at,self.stop+1):
                segment,_ = next(rounds)
                meter.update(segment)
                if t == self.at:
                    h = float(segment['h_strategy'][-1])
                if t <= pulse_end:
                    pulse_d.append(segment['loss_degree'][-1])
            outcomes[arm] = probe_outcomes(c,self.at,meter.events,h,math.fsum(pulse_d)/span)
            if arm=='control' and expected is not None:
                for key,value in expected.items():
                    match = (math.isclose(outcomes[arm][key],value,rel_tol=1e-10,abs_tol=1e-12)
                             if key in ('h','pulse_loss') else outcomes[arm][key]==value)
                    if not match:
                        return {'at':self.at,'reason':'대조 분기가 원자료와 일치하지 않아 이 짝을 제외했어.'}
        if outcomes['crowd']['pulse_loss']!=outcomes['loss_shield']['pulse_loss']:
            raise AssertionError('패점 보전 분기의 선택 기록이 달라졌어.')
        return {'at':self.at,'pulse_end':pulse_end,'first_clean_v':first_clean,
                'horizon_end':horizon_end,'members':len(members),'option':option,
                'arms':outcomes}


def matched_event_results(c, folder, mode, runs):
    """동일 방식·동일 라운드의 다른 회차. 노출 사건마다 대조군을 평균해 한 표를 준다."""
    events = []
    for run in runs:
        groups = {key:array('Q') for key in ('H_strategy','L','V')}
        path = folder/mode/f'run_{run:02d}_events.csv'
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                key = row['event']
                if key in groups and int(row['confirmed_round'])<=c.rounds:
                    groups[key].append(int(row['start_round']))
        for times in groups.values():
            if any(b<=a for a,b in zip(times,islice(times,1,None))):
                raise ValueError(f'{path.name}의 사건 시작 라운드 순서를 확인해 줘.')
        events.append(groups)
    delay_l = c.loss_block*c.loss_repeats-1
    delay_v = c.vol_hold-1
    def next_start(times,t):
        i = bisect_right(times,t)
        return times[i] if i<len(times) else c.rounds+1
    results = {}
    for key, source, lag, pad in (('HL','H_strategy',0,delay_l),('HLV','H_strategy',0,max(delay_l,delay_v)),
                                   ('LV','L',delay_l,delay_v)):
        anchors = merge(*(zip(e[source],repeat(i)) for i,e in enumerate(events)))
        matched = exposed_hits = no_control = short = 0
        other_hits = 0.
        for t,group in groupby(anchors,lambda a:a[0]):
            exposed = {i for _,i in group}
            k = len(exposed)
            begin, end = t+lag, t+lag+c.chain_horizon
            if end+pad>c.rounds:
                short += k
                continue
            if k==len(events):
                no_control += k
                continue
            yes = no = 0
            for i,e in enumerate(events):
                if key=='LV':
                    hit = next_start(e['V'],begin)<=end
                else:
                    loss = next_start(e['L'],begin)
                    hit = loss<=end and (key=='HL' or next_start(e['V'],loss)<=end)
                if i in exposed:
                    yes += hit
                else:
                    no += hit
            matched += k
            exposed_hits += yes
            other_hits += k*(no/(len(events)-k))
        results[key] = {'n':matched,'hits':exposed_hits,'control_sum':other_hits,
                        'unmatched':no_control,'short':short}
    return results


def paired_binary_p(plus, minus):
    """쌍별 불일치 수의 정확 양측 이항 검정. 동률을 포함한 양쪽 꼬리를 합한다."""
    n = plus+minus
    k = min(plus,minus)
    if plus==minus:
        return 1.
    log_mass = math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)
    term = mass = 1.
    for j in range(k,0,-1):
        term *= j/(n-j+1)
        mass += term
        if term < 1e-16*mass:
            break
    return min(1., 2*math.exp(log_mass+math.log(mass)))


EDGE_NAMES = {'HL':'H↓ → L','LV':'L → V↑','HLV':'H↓ → L → V↑'}


def evidence_panel(c, runs, observed, trials):
    """웹과 저장 HTML에 같은 관측 지표와 회차 단위 짝비교를 표시한다."""
    def pct(x):
        return f'{100*x:.2f}%'
    def pp(x):
        return f'{100*x:+.2f}%p'
    def table(headers, body):
        head = ''.join(f'<th scope="col">{s}</th>' for s in headers)
        return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    def row(m, values):
        return f'<tr><th scope="row" class="{m}">{MODE_NAMES[m]}</th>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>'
    obsrows, effects, checks, missing = [], [], [], []
    for mode in MODES:
        for key in ('HL','LV','HLV'):
            r = observed[mode][key]
            n = r['n']
            total = n+r['unmatched']+r['short']
            if n:
                on,off = r['hits']/n,r['control_sum']/n
                vals = [str(n),pct(on),pct(off),pp(on-off)]
            else:
                why = '선행 사건 미관측' if total==0 else ('대조 회차 없음' if r['unmatched'] else '추적 기간 부족')
                vals = ['0',why,'—','—']
            obsrows.append(row(mode,[EDGE_NAMES[key],*vals,str(r['unmatched']),str(r['short'])]))
        records = [trials[r][mode] for r in runs if r in trials and mode in trials[r]]
        arms = [rec['arms'] for rec in records if 'arms' in rec]
        omitted = Counter(rec.get('reason','기록 없음') for rec in records if 'arms' not in rec)
        if len(records)<len(runs):
            omitted['개입 기록 없음'] += len(runs)-len(records)
        if omitted:
            missing.append(MODE_NAMES[mode]+' · '+', '.join(f'{html.escape(why)} ({n}회차)' for why,n in omitted.items()))
        if not arms:
            why = next((rec['reason'] for rec in records if rec.get('reason')),
                       '이 결과 폴더에는 개입 실험 기록이 없어.')
            effects.append(row(mode,['개입 결과', '0',html.escape(why),'—','—','—','—','—','—']))
            continue
        for arm,quantity,flag,label in (('homogeneous','h','h_event','전략 통일 · H (음수면 감소)'),
                                        ('crowd','pulse_loss','l_event','선택 쏠림 · 1인당 손실 (양수면 증가)')):
            delta = sum(a[arm][quantity]-a['control'][quantity] for a in arms)/len(arms)
            regular = sum(a['control'][flag] for a in arms)
            changed = sum(a[arm][flag] for a in arms)
            checks.append(row(mode,[label,str(len(arms)),f'{delta:+.5f}',f'{changed}/{len(arms)}',f'{regular}/{len(arms)}']))
        contrasts = (('HL','homogeneous','control','전략 통일 → L',True),
                     ('LV','crowd','loss_shield','패점 반영 → 후속 V↑',True),
                     ('LV','crowd','control','선택 쏠림 → 후속 V↑',False),
                     ('HLV','homogeneous','control','전략 통일 → L 뒤 V↑',False))
        for key,arm,reference,label,primary in contrasts:
            pairs = [(a[arm][key],a[reference][key]) for a in arms if reference in a and a[arm][key] is not None and a[reference][key] is not None]
            n = len(pairs)
            if not n:
                effects.append(row(mode,[label,'0','추적 기간 부족','—','—','—','—','—','—']))
                continue
            plus = sum(on and not off for on,off in pairs)
            minus = sum(off and not on for on,off in pairs)
            on = sum(a for a,_ in pairs)/n
            off = sum(b for _,b in pairs)/n
            if not primary:
                p_text,meaning = '—','보조 비교 · 기술통계'
            else:
                adj = min(1.,4*paired_binary_p(plus,minus))
                p_text = f'{adj:.4g}' if n>1 else '—'
                shifted = (sum(a['homogeneous']['h']-a['control']['h'] for a in arms)<0 if key=='HL'
                           else sum(a['crowd']['pulse_loss']-a['control']['pulse_loss'] for a in arms)>0)
                if n<2:
                    meaning='독립 반복 필요'
                elif not shifted:
                    meaning='선행 지표의 의도한 변화 미확인'
                elif adj>=.05:
                    meaning='개입 효과의 차이 불확실'
                else:
                    meaning='개입 후 발생 증가' if on>off else '개입 후 발생 감소'
                    flag = 'l_event' if key=='LV' else 'h_event'
                    if sum(a[arm][flag]-a['control'][flag] for a in arms)<=0:
                        meaning += ' · 선행 사건 증가 미확인'
            wait = 'wait_'+key
            diffs = [a[arm][wait]-a[reference][wait] for a in arms
                     if reference in a and a[arm].get(wait) is not None and a[reference].get(wait) is not None]
            waiting = f'{sum(diffs)/len(diffs):+.1f}' if diffs else '—'
            effects.append(row(mode,[label,str(n),pct(on),pct(off),pp(on-off),waiting,f'{plus} / {minus}',p_text,meaning]))
    at,_,_ = probe_schedule(c)
    span = c.loss_block*c.loss_repeats
    return ('<h2>선행 사건이 후속 사건에 영향을 주었을까?</h2>'
        '<h3>같은 라운드의 관측 비교</h3>'
        f'<p>선행 사건이 시작된 시점과 같은 방식의 다른 회차에서 그 사건이 시작되지 않은 같은 라운드를 비교해. '
        f'H↓는 발생 직후부터, L은 확인된 직후부터 {c.chain_horizon:,}라운드를 추적해. '
        '대조 회차들의 비율을 사건별로 먼저 평균해. 추적이 끝나지 않은 사건과 대조가 없는 사건은 분모에서 빼.</p>'
        +table(['방식','후속 사건','비교 사건','선행 사건 있음','대조','차이','대조 없음','기간 부족'],obsrows)
        +'<p class="helper">관측 비율 차이는 예측과 관련돼. 가까운 시점은 같은 후속 사건을 공유할 수 있어.</p>'
        +'<h3>같은 상태에서 개입한 짝비교</h3>'
        +f'<p>회차마다 {at:,}라운드 직전의 전략·점수·난수 상태를 복제해 네 갈래로 진행해. 대조군은 원래 규칙을 따르고, '
        f'전략 통일군은 무작위로 뽑은 {max(2,c.n//2):,}명의 전략을 그들의 평균 확률 벡터로 한 번 바꿔. 전체의 선택지별 평균 확률은 보존돼. '
        f'선택 쏠림군은 같은 인원에게 무작위로 정한 한 선택지를 {span}라운드 동안 선택하게 해. 패점 보전군은 쏠림군의 개인별 선택과 승자를 그대로 재생하고, 이 기간 패자의 점수 차감만 0으로 바꿔. 이후에는 모두 원래 규칙으로 돌아가.</p>'
        +f'<p>‘패점 반영’의 대조 열은 패점 보전군이고 다른 행의 대조 열은 원래 게임이야. 전략 통일 직후부터 {c.chain_horizon:,}라운드 안에 새 L 또는 L 뒤 V↑가 시작됐는지 세어. '
        f'쏠림 실험의 V↑는 개입 마지막 라운드에서 {2*c.vol_window}라운드가 지난 때부터 같은 종료 시점까지 세어. '
        '이렇게 하면 진동의 두 측정 창에 강제로 선택한 라운드가 들어가지 않아.</p>'
        +table(['방식','측정 경로','독립 짝','개입','대조','차이','평균 대기 차이 (라운드)','개입만 / 대조만','보정 p','해석'],effects)
        +'<p class="helper">대기 차이는 개입군에서 대조군을 뺀 값이야. 음수면 더 일찍 나타난 거야. 기간 내 사건이 없으면 마지막 관측 다음 라운드까지로 계산해.</p>'
        +('<p class="helper">'+'<br>'.join(missing)+'</p>' if missing else '')
        +('<details><summary>개입이 목표 지표를 바꿨는지 확인</summary>'
          +table(['방식','개입과 지표','독립 짝','평균 개입−대조','개입군 사건','대조군 사건'],checks)
          +'<p>전략 엔트로피는 개입 첫 라운드, 1인당 손실은 쏠림 기간 평균이야. H↓는 개입 첫 시점, L은 쏠림 기간에 시작된 사건을 세어.</p></details>' if checks else '')
        +'<p class="note">회차가 독립 짝의 단위야. 두 기본 경로와 두 모드의 총 네 검정에 정확 양측 맥니마 검정과 본페로니 보정(기준 0.05)을 적용해. '
        '목표 지표가 실제로 움직였는지와 관계없이 모든 유효 짝을 포함해. '
        '개입 결과는 이 게임에서 전략 통일, 선택 쏠림 또는 패점 반영 규칙을 바꾼 효과를 추정해. 세 지표 사이의 직접 인과나 매개효과는 이 실험만으로 구별할 수 없어.</p>')


def write_evidence(folder,c,runs,output=None,rebuilt=None):
    output = folder if output is None else output
    observed = {m:matched_event_results(c,folder,m,runs) for m in MODES}
    trials = dict(rebuilt or {})
    for r in runs:
        path = folder/f'run_{r:02d}_interventions.json'
        if r not in trials and path.is_file():
            trials[r] = json.loads(path.read_text(encoding='utf-8'))
    panel = evidence_panel(c,runs,observed,trials)
    body = '<main><header><h1>관측 대조와 개입 실험 결과</h1></header><section>'+panel+'</section></main>'
    write_atomic_text(output/'chain_evidence.html',shell('관측 대조와 개입 실험 결과',body))
    values = {'version':VERSION,'runs':runs,'config':asdict(c),'observed':observed,'trials':trials}
    write_atomic_text(output/'chain_evidence.json',json.dumps(values,ensure_ascii=False,allow_nan=False))
    return panel


def archive_source(path):
    source = Path(path).expanduser().resolve()
    if not (source/'config.json').is_file() and (source/'results'/'config.json').is_file():
        source = source/'results'
    if not (source/'config.json').is_file():
        raise ValueError('압축을 풀고 config.json이 있는 결과 폴더의 경로를 입력해 줘.')
    return source


def replay_archived_pair(c,mode,run,use_numpy,source):
    meter = Measurements(c)
    study = InterventionPair(c,mode,run,meter)
    if study.reason:
        return {'at':study.at,'reason':study.reason}
    file = source/mode/f'run_{run:02d}_trace.csv'
    if not file.is_file():
        return {'reason':'원래 경로를 검증할 시계열 CSV가 없어.'}
    cap = 2+max(c.h_window,c.vol_window,c.loss_base+c.loss_block*c.loss_repeats)
    tail = {k:TailColumn(cap) for k in (*TRACE_KEYS,'counts')}
    generate = rounds_numpy if use_numpy else rounds_python
    replay = generate(replace(c,rounds=study.stop),mode,run,False,False,tail,checkpoint=study)
    span = c.loss_block*c.loss_repeats
    pulse_loss = []
    columns = [f'count_{j+1}' for j in range(c.q)]
    with file.open(encoding='utf-8-sig',newline='') as f:
        rows = csv.DictReader(f)
        for t in range(1,study.stop+1):
            row = next(rows,None)
            tr,_ = next(replay)
            meter.update(tr)
            same = (row is not None and int(row['round'])==t
                    and all(int(row[k])==n for k,n in zip(columns,tr['counts'][-1]))
                    and int(row['replacements'])==tr['replacements'][-1]
                    and int(row['mutations'])==tr['mutations'][-1]
                    and math.isclose(float(row['gain']),tr['gain'][-1],abs_tol=1e-10)
                    and all(math.isclose(float(row['H_'+k]),tr['h_'+k][-1],abs_tol=1e-10)
                            for k in ENTROPIES))
            if not same:
                return {'at':study.at,'reason':f'{t:,}라운드에서 저장 기록과 재생 경로가 달라 개입 비교를 제외했어.'}
            if t<study.at:
                for g in meter.events.values():
                    g.clear()
            if t==study.at:
                h = tr['h_strategy'][-1]
            if study.at<=t<study.at+span:
                pulse_loss.append(tr['loss_degree'][-1])
    expected = probe_outcomes(c,study.at,meter.events,h,math.fsum(pulse_loss)/span)
    result = study.calculate(use_numpy,expected)
    result['source'] = 'verified_replay'
    return result


def analyze_archive(source,output,on_progress=None):
    meta = json.loads((source/'config.json').read_text(encoding='utf-8-sig'))
    c = Config(**meta['config'])
    validate(c)
    runs = sorted(int(p.name[4:-11]) for p in (source/'random').glob('run_*_events.csv')
                  if p.name[4:-11].isdigit() and (source/'best'/p.name).is_file())
    summary = source/'summary.csv'
    if summary.is_file():
        with summary.open(encoding='utf-8-sig',newline='') as f:
            completed = {(int(row['run']),row['mode']) for row in csv.DictReader(f)}
        runs = [r for r in runs if all((r,m) in completed for m in MODES)]
    else:
        runs = [r for r in runs if all((source/m/f'run_{r:02d}_analysis.json').is_file() for m in MODES)]
    if not runs:
        raise ValueError('완료된 회차의 랜덤·베스트 결과 파일을 찾을 수 없어.')
    label = meta.get('engine','')
    use_numpy = label.startswith('NumPy')
    reason = None
    if use_numpy:
        try:
            import numpy
        except (ImportError,OSError):
            reason = '이 기록의 재생에 NumPy가 필요해.'
    elif not label.startswith('Python'):
        reason = '기록에 원래 난수 엔진이 표시되어 있지 않아.'
    reconstructed = {}
    for pos,run in enumerate(runs,1):
        if on_progress:
            on_progress(f'{pos} / {len(runs)}회차 확인 중')
        if (source/f'run_{run:02d}_interventions.json').is_file():
            continue
        reconstructed[run] = {}
        for m in MODES:
            if reason:
                result = {'reason':reason}
            else:
                try:
                    result = replay_archived_pair(c,m,run,use_numpy,source)
                except (KeyError,ValueError,OSError,StopIteration) as exc:
                    result = {'reason':'저장 기록을 재생할 수 없어. '+str(exc)}
            reconstructed[run][m] = result
    if on_progress:
        on_progress('같은 라운드의 대조 회차 비교 중')
    write_evidence(source,c,runs,output,reconstructed)
    return output/'chain_evidence.html'


def result_intro(c):
    return (f'<p class="result-intro">전략 분포 엔트로피를 기준으로 첫 <b>엔트로피 감소(H↓) → 집단 패배(L) → 진동 증폭(V↑)</b>가 '
            f'이 순서로 발생하고 H↓부터 V↑까지 {c.chain_horizon:,}라운드 이내면 연쇄 충족이야. '
            f'H↓는 직전 {c.h_window}라운드 평균보다 엔트로피가 {c.h_drop:g} 이상 낮아진 사건이야. 필요한 사건이 미관측이면 판정을 보류해.</p>')


def completion_content(c, per_run, aggregate, live=False, evidence=None):
    runs = sorted(per_run)
    latest = f'run-{runs[-1]}'
    automatic = 'overall' if len(runs)==c.repeats and len(runs)>1 else latest
    default = 'overall' if len(runs)>1 else latest
    if live:
        default = automatic
    panels, picker = [], ''
    if len(runs)>1 or live:
        opts = [('auto','최근 완료 결과 자동 표시')] if live else []
        opts.append(('overall',f'완료 회차 종합 ({len(runs)}회)'))
        opts.extend((f'run-{r}',f'{r}회차') for r in runs)
        selected = 'auto' if live else default
        options = ''.join(f'<option value="{k}"'+(' selected' if k==selected else '')+f'>{v}</option>' for k,v in opts)
        picker = f'<label class="result-picker">결과 선택 <select data-result-choice data-auto="{automatic}">{options}</select></label>'
        panels.append(('overall',overall_result(c,per_run,aggregate)))
    for r in runs:
        panel = run_result(c,r,per_run[r])
        if live:
            panel += f'<a href="/files/compare_{r:02d}.html" target="_blank" rel="noopener">이 회차의 상세 그래프 열기 ↗</a>'
        panels.append((f'run-{r}',panel))
    views = ''.join(f'<div data-result-view="{name}"'+('' if name==default else ' hidden')+f'>{body}</div>'
                    for name,body in panels)
    download = '<p><a href="/files/chain_summary.html" download>완료 회차 종합 결과 HTML 저장</a></p>' if live else ''
    prefix = '/files/' if live else ''
    more = ('<p><a href="'+prefix+'chain_evidence.html">관측 대조와 개입 실험 결과 열기</a></p>')
    if evidence:
        more = '<div id="chain-effects">'+evidence+'</div><p><a href="'+prefix+'chain_evidence.json" download>대조·개입 결과 수치 저장</a></p>'
    return (f'<div class="section-head"><h2>연쇄관계 종합 결과</h2>{picker}</div>'+result_intro(c)+views+more+download)


def result_section(content):
    display = '' if content else ' hidden'
    return f'<section id="completion" aria-label="연쇄관계 종합 결과" tabindex="-1"{display}>{content}</section>'


def write_atomic_text(path, content):
    pending = path.with_suffix('.pending')
    pending.write_text(content,encoding='utf-8')
    pending.replace(path)


def report_pair(folder, c, run, pair, engine, use_numpy):
    if pair.round != c.rounds:
        raise ValueError('두 방식이 마지막 라운드까지 진행되면 종합 결과를 저장할 수 있어.')
    analysis, finished_sides = {}, {}
    sources = {}
    summaries = []
    for mode in MODES:
        tr, events = pair.traces[mode], pair.meters[mode].events
        stem = folder / mode / f'run_{run:02d}'
        if pair.tables:
            pair.tables[mode].flush()
            write_large_trace(stem,pair.tables[mode],c)
        else:
            write_trace(stem.with_name(stem.name+'_trace.csv'), tr, c)
        with stem.with_name(stem.name+'_events.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['event', 'start_round', 'confirmed_round'])
            for key, group in events.items():
                for e in group:
                    w.writerow(['H_'+key if key in ENTROPIES else key, e['t'], e['confirmed']])
        if pair.tables:
            windows = disk_windows(pair.tables[mode],events['L'],c)
        else:
            by_time = window_changes(tr,events['L'],c)
            windows = ((e['t'],*by_time.get(str(e['t']),(None,None))) for e in events['L'])
        ratios = []
        with stem.with_name(stem.name+'_loss_windows.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['loss_start', 'pre_rms', 'post_rms', 'ratio_post_pre'])
            for t,before,after in windows:
                ratio = after/before if before and after is not None else None
                w.writerow([t,before,after,ratio])
                if ratio is not None:
                    ratios.append(ratio)
        phases = {}
        for phase in PHASES:
            start, end = phase_range(c, phase, pair.round)
            common = common_events = None
            phases[phase] = {}
            for kind in ENTROPIES:
                curves = phase_curves(tr, c, phase, kind, use_numpy, common)
                flags = phase_curves(tr, c, phase, kind, use_numpy, common_events, events)
                common, common_events = curves['L_V'], flags['L_V']
                summary = phase_summary(events, c, start, end, kind)
                phases[phase][kind] = {'summary': summary, 'curves': curves, 'event_curves': flags}
                first = summary['first']
                summaries.append({'run': run, 'mode': mode, 'phase': phase, 'entropy': kind,
                    **{'t_'+k: v['t'] if v else None for k, v in first.items()},
                    **{k: summary[k] for k in ('HL','LV','chain')}})
        write_lag_csv(stem.with_name(stem.name+'_lag.csv'), phases)
        analysis[mode] = phases
        finished_sides[mode] = completed_side(c,events,phases,ratios)
        if pair.tables:
            write_large_analysis(stem.with_name(stem.name+'_analysis.json'),events,phases)
            dump_json(stem.with_name(stem.name+'_overview.json'),{'phases':phases,'side':finished_sides[mode]})
            sources[mode] = {'series':{},'events':{k:list(g.starts) for k,g in events.items()}}
            continue
        dump_json(stem.with_name(stem.name+'_analysis.json'), {'events': events, 'phases': phases})
        # 전체 라운드를 보존한다. 표시에만 12번째 유효숫자에서 반올림한다.
        series = {k: [None if v is None else float(format(v, '.12g')) for v in tr[k]] for k in SERIES}
        series['counts'] = tr['counts']
        sources[mode] = {'series': series, 'events': events}
    studies = {m:pair.studies[m].calculate(use_numpy,recorded_probe(c,pair.traces[m],pair.meters[m].events))
               for m in MODES}
    write_atomic_text(folder/f'run_{run:02d}_interventions.json',json.dumps(studies,ensure_ascii=False,allow_nan=False))
    data = {'config': asdict(c), 'engine': engine, 'run': run, 'round': pair.round,
            'modes': sources, 'curves': analysis, 'large': bool(pair.tables)}
    target = folder/f'compare_{run:02d}.html'
    single = completion_content(c,{run:finished_sides},None)
    write_atomic_text(target, dashboard(c, data, live=False, completion=single))
    if pair.tables:
        pair.final_phases = analysis
    return summaries


def write_index(folder, c, rows, engine, live=False):
    with (folder/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    runs = sorted({r['run'] for r in rows})
    safe_engine = html.escape(engine)
    links = ' '.join(f'<a class="pill" href="compare_{r:02d}.html">{r}회차 비교 ↗</a>' for r in runs)
    table = []
    aggregate = {phase: {kind: {'curves': {}, 'event_curves': {}} for kind in ENTROPIES} for phase in PHASES}
    completed_runs = {r: {} for r in runs}
    for mode in MODES:
        reports = []
        for run in runs:
            stem = folder/mode/f'run_{run:02d}'
            summary_path = stem.with_name(stem.name+'_overview.json')
            if summary_path.is_file():
                small = json.loads(summary_path.read_text(encoding='utf-8'))
                reports.append(small['phases'])
                completed_runs[run][mode] = small['side']
                continue
            doc = json.loads(stem.with_name(stem.name+'_analysis.json').read_text(encoding='utf-8'))
            reports.append(doc['phases'])
            with stem.with_name(stem.name+'_loss_windows.csv').open(encoding='utf-8-sig', newline='') as f:
                ratios = [float(row['ratio_post_pre']) for row in csv.DictReader(f) if row['ratio_post_pre']]
            completed_runs[run][mode] = completed_side(c,doc['events'],doc['phases'],ratios)
        for phase in PHASES:
            for kind in ENTROPIES:
                for basis in ('curves','event_curves'):
                    curves = {}
                    for key, _, _ in PAIRS:
                        per_run = [p[phase][kind][basis][key] for p in reports]
                        curve = []
                        for group in zip(*per_run):
                            valid = [item['r'] for item in group if item['r'] is not None]
                            curve.append({'lag': group[0]['lag'], 'r': sum(valid)/len(valid) if valid else None,
                                          'n_runs': len(valid)})
                        curves[key] = curve
                    aggregate[phase][kind][basis][mode] = curves
    with (folder/'mean_lag.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['phase','entropy','basis','mode','pair','lag','mean_r','n_runs'])
        for phase in PHASES:
            for kind in ENTROPIES:
                for basis in ('curves','event_curves'):
                    for mode, curves in aggregate[phase][kind][basis].items():
                        for key, curve in curves.items():
                            writer.writerows((phase,kind,'continuous' if basis=='curves' else 'events',mode,key,v['lag'],v['r'],v['n_runs']) for v in curve)
    for phase in PHASES:
        for kind in ENTROPIES:
            for mode in MODES:
                subset = [r for r in rows if r['mode']==mode and r['phase']==phase and r['entropy']==kind]
                known = sum(r['chain'] is not None for r in subset)
                satisfied = sum(r['chain'] is True for r in subset)
                intervals = []
                for k in ('HL','LV'):
                    values = [r[k] for r in subset if r[k] is not None]
                    intervals.append(format(median(values), ',g') if values else '—')
                cells = (PHASE_NAMES[phase], HLABELS[kind], MODE_NAMES[mode], len(subset), known,
                         satisfied, *intervals)
                table.append(f'<tr data-phase="{phase}" data-kind="{kind}">'+
                             ''.join('<td>'+html.escape(str(x))+'</td>' for x in cells)+'</tr>')
    if len(runs)==c.repeats:
        evidence = write_evidence(folder,c,runs)
    else:
        evidence = None
        pending = (f'<main><h1>관측 대조와 개입 실험</h1><p>{len(runs)} / {c.repeats}회차 완료. '
                   '마지막 회차가 끝나면 같은 라운드의 대조군과 독립 개입 짝을 함께 집계해.</p></main>')
        write_atomic_text(folder/'chain_evidence.html',shell('관측 대조와 개입 실험',pending))
    overview = completion_content(c,completed_runs,aggregate,evidence=evidence)
    overview_block = result_section(overview)
    summary_body = (f'<main><header><div class="eyebrow">진화 소수파 게임</div><h1>실험 완료 보고서</h1>'
                    f'<p>{c.n:,}명 · 선택지 {c.q}개 · 회차당 {c.rounds:,}라운드 · {len(runs)}/{c.repeats}회차 완료</p>'
                    '</header>'+overview_block+'<details><summary>H↓·L·V↑의 판정 기준</summary>'+methods(c)+'</details></main>')
    summary_script = COMPLETION_UI+"\nselectCompletedView(document.getElementById('completion'));"
    write_atomic_text(folder/'chain_summary.html',shell('연쇄관계 종합 결과',summary_body,summary_script))
    body = f"""<main><header><div class="eyebrow">2026 영재원 연구 · 진화 소수파 게임</div><h1>반복 실험 결과</h1>
<p>{c.n}명 · {c.q}개 선택지 · {c.rounds:,}라운드 · {len(runs)}/{c.repeats}회차 완료 · {safe_engine}</p>
<div class="key"><span class="random">랜덤 · 무작위 참가자 모방</span><span class="best">베스트 · 최고점자 모방</span></div></header>
{overview_block}<section><h2>회차별 비교 화면</h2><p>두 방식의 시계열과 사건 시점, 패배 전후 변화, 시차 상관을 한 화면에서 볼 수 있어.</p><div class="actions">{links}</div></section>
<div class="analysis-filter"><label>분석 구간 <select id="phase"><option value="all">전체</option><option value="early">초기 적응기</option><option value="late">적응 이후</option></select></label>
<label>엔트로피 <select id="kind"><option value="strategy">전략 분포</option><option value="lineage">계보 분포</option></select></label></div>
<section><h2>반복 간 사건 순서</h2><div class="scroll"><table><thead><tr><th>구간</th><th>엔트로피</th><th>방식</th><th>완료 회차</th><th>세 사건 관측</th><th>순서·간격 충족</th><th>L−H 중앙값</th><th>V−L 중앙값</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="note">간격은 라운드 단위야. 각 간격의 중앙값은 해당 두 사건이 모두 관측된 회차에서 구해.</p>
<a href="summary.csv" download>회차별 사건 요약 CSV</a></section>
<section><h2>회차별 시차 상관의 평균</h2><label>상관 분석 대상 <select id="lagBasis"><option value="continuous">측정값 E · D · A</option><option value="events">사건 발생 여부 (0·1)</option></select></label><p>각 회차의 r을 먼저 계산한 뒤 같은 시차끼리 산술평균해. 각 회차에 같은 가중치를 주고, r을 구할 수 없는 회차는 해당 시차에서 제외해.</p>
<div id="mean-H_L" class="plot"></div><div id="mean-L_V" class="plot"></div><div id="mean-H_V" class="plot"></div>
<a href="mean_lag.csv" download>평균 상관계수와 유효 회차 수 CSV</a></section>{methods(c)}</main>"""
    script = 'const GROUP='+wire_json(aggregate)+';\n'+CHART+'\n'+SUMMARY_UI
    write_atomic_text(folder/'index.html',shell('소수파 게임 · 반복 요약',body,summary_script+'\n'+script))
    if live:
        return completion_content(c,completed_runs,aggregate,live=True,evidence=evidence)

STYLE = r"""
:root{--ink:#172d43;--muted:#516579;--line:#d8e1e9;--blue:#1769bd;--orange:#ba510e;--bg:#f3f6f8}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Inter","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;word-break:keep-all}main{width:min(1152px,100%);margin:auto;padding:36px 24px 60px}header{padding:18px 0 12px}h1,h2,h3{letter-spacing:-.045em;line-height:1.35;margin:0}h1{font-size:clamp(30px,4.2vw,48px);margin:.22em 0}h2{font-size:clamp(20px,2.7vw,25px)}h3{font-size:17px}p{margin:.65em 0 1em}.eyebrow,.helper,.section-head>span,small{color:var(--muted)}.eyebrow{font-size:13px;letter-spacing:.06em;font-weight:650}.mast,.section-head,.actions,.analysis-filter,.key,nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.mast,.section-head{justify-content:space-between}.key{gap:24px}.key span{display:inline-flex;align-items:center;gap:7px;font-weight:720}.key span:before{content:"";width:30px;border-top:3px solid currentColor}.key .best:before{border-top-style:dashed}.random{color:var(--blue)}.best{color:var(--orange)}#clock,td,code,.hover{font-variant-numeric:tabular-nums}.section-head{margin-bottom:12px}nav{padding:16px 0 12px;gap:20px;border-bottom:1px solid var(--line)}a{color:#16538a;text-underline-offset:4px}nav a{font-size:14px;font-weight:630;text-decoration:none}.analysis-filter{margin:16px 0}section{background:#fff;border:1px solid var(--line);border-radius:15px;padding:24px 26px;margin:15px 0}button,select,input{font:inherit;border:1px solid #bacbd8;border-radius:7px;background:#fff;color:var(--ink);padding:6px 9px}button,.pill,summary{cursor:pointer}button:hover,select:hover,.pill:hover{background:#f0f5f9}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #60a5e0;outline-offset:2px}button:disabled{opacity:.45;cursor:default}button.primary,.pill{color:white;background:#1c4266;border-color:#1c4266;font-weight:650}button.primary:hover,.pill:hover{background:#2d557b}.pill{border-radius:6px;padding:6px 12px;text-decoration:none}.actions label,.analysis-filter label{font-size:13px;color:var(--muted)}.actions select,.analysis-filter select{color:var(--ink)}input[type=number]{width:95px}.progress{height:5px;background:#e6edf3;border-radius:10px;margin:15px 0}.progress div{height:100%;background:#416786;border-radius:10px}#error{color:#9c361b}.helper,.hover,.note{font-size:13px}.note{background:#f2f6fa;padding:12px 14px;border-radius:7px;color:#40556c}.helper{margin:9px 0}button,.pill,select{touch-action:manipulation}details{margin:12px 0}summary{font-weight:650}fieldset{border:0;padding:8px 0;margin:0}.settings-grid,.worlds,.definitions-grid{display:grid;gap:14px}.settings-grid{grid-template-columns:repeat(auto-fit,minmax(135px,1fr));margin:12px 0}.settings-grid input{display:block;width:100%}.worlds{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:14px}.world{border:1px solid var(--line);border-top:3px solid currentColor;border-radius:8px;padding:12px}.world small{font-size:12px;font-weight:450;margin-left:5px}.world canvas{display:block;width:100%;height:168px}.world p{margin:5px 0 0;color:var(--muted)}.scroll{overflow-x:auto;margin-top:12px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}th,td{text-align:right;padding:10px 9px;border-bottom:1px solid var(--line)}th{background:#f2f6f9;color:#3e536a;font-size:12px}th:first-child,td:first-child{text-align:left}td small{display:block;font-size:11px}.plot{margin:18px 0 8px}.plot svg{width:100%;height:auto;display:block}.plot-bottom{display:flex;justify-content:space-between;gap:10px;align-items:center}.plot-bottom button{font-size:12px;padding:3px 8px}.hover{color:var(--muted)}.definitions-grid{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:16px}.definitions article{background:#f5f8fa;border-radius:8px;padding:15px}.definitions article p{font-size:14px}.symbol{font-size:13px;font-weight:700;color:#4d6680}.definitions code{display:block;white-space:normal;font:13px/1.7 ui-monospace,monospace;padding:8px;background:#e8eff4;border-radius:5px}.empty{padding:20px;color:var(--muted);margin:0}#reports:empty::after{content:"완료한 회차가 여기에 표시돼.";color:var(--muted)}[hidden]{display:none!important}@media(max-width:720px){main{padding:12px 10px 32px}section{padding:17px 12px}.definitions-grid,.worlds{grid-template-columns:1fr}.actions,.analysis-filter{gap:7px}nav{gap:13px}.world canvas{height:145px}}@media print{body{background:white}button,nav,#controls{display:none}.plot,article,tr{break-inside:avoid}section{border:0;padding:12px 0}}

#completion{border:1px solid #c0ced9;scroll-margin-top:12px}#completion[hidden],[data-result-view][hidden]{display:none!important}.result-intro{max-width:82ch;color:#415568}.result-picker{font-size:14px;color:#415568}.result-picker select{max-width:100%}.result-subtitle{font-size:15px;color:#375269;margin:18px 0 10px;letter-spacing:-.015em}.result-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.result-card{padding:16px 18px;border-radius:10px;border-left:4px solid currentColor;background:#f1f6fb}.result-card.best{background:#fcf4ed}.result-card .method-name{margin:0 0 5px;color:inherit;font-size:13px;font-weight:700}.result-card h3{color:var(--ink);font-size:clamp(20px,2.3vw,26px)}.result-card p{margin:8px 0 0;font-size:14px;color:#3f5265}.result-card .later{border-top:1px solid var(--line);padding-top:8px}.milestones{margin:14px 0 8px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.milestones dt{font-size:12px;color:#4e6275}.milestones dd{margin:3px 0;color:var(--ink);font-weight:700;font-variant-numeric:tabular-nums}.result-table th[scope=row]{font-size:14px;background:transparent;text-align:left;font-weight:550;white-space:normal}.result-table td{width:25%}.result-evidence{margin:14px 1px 0}.result-evidence p{font-size:13px;color:#4b6074}.result-evidence h4{margin:14px 0 4px}@media(max-width:650px){.result-cards{grid-template-columns:1fr;gap:9px}.result-card{padding:12px}.result-table th,.result-table td{font-size:12px;padding:8px 5px}}
"""

COMPLETION_UI = r"""
// 한 번 완료한 회차의 결과는 실시간 그래프의 회차·라운드와 독립적으로 보관한다.
function selectCompletedView(host, previousChoice){
 const select=host.querySelector('[data-result-choice]');
 if(!select)return;
 if(previousChoice&&Array.from(select.options).some(o=>o.value===previousChoice))select.value=previousChoice;
 function apply(){
  const view=select.value==='auto'?select.dataset.auto:select.value;
  for(const panel of host.querySelectorAll('[data-result-view]'))panel.hidden=panel.dataset.resultView!==view;
 }
 select.onchange=apply;apply();
}
function receiveCompletedResult(state){
 const host=document.getElementById('completion'),count=state.completed.length;
 if(!state.completion_html||Number(host.dataset.completed||0)===count)return;
 const prior=host.querySelector('[data-result-choice]')?.value;
 const reveal=host.hidden||((!prior||prior==='auto')&&count===state.config.repeats);
 host.innerHTML=state.completion_html;host.hidden=false;host.dataset.completed=String(count);
 selectCompletedView(host,prior);
 if(reveal)host.scrollIntoView({block:'start',behavior:'auto'});
}
"""

CHART = r"""
const $ = id => document.getElementById(id);
const modes = ['random','best'];
const colors = {random:'#1769bd',best:'#ba510e'};
const names = {random:'랜덤',best:'베스트'};
const letters = {H:'H↓',L:'L',V:'V↑'};
const numeric = x => typeof x === 'number' && Number.isFinite(x);
const nf = (x, digits=3) => numeric(x) ? x.toLocaleString('ko-KR',{maximumFractionDigits:digits}) : '—';
const signed = (x,d=0) => numeric(x) ? (x>0?'+':'')+nf(x,d) : '—';
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function saveBlob(blob,name){
 const a=document.createElement('a'),url=URL.createObjectURL(blob);
 a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),2000);
}
function chart(id,title,xs,series,opts={}){
 const host=$(id), W=Math.max(370,Math.min(1060,host.clientWidth||900)),H=247;
 const L=W<520?58:73,R=18,T=60,B=47,PW=W-L-R,PH=H-T-B;
 const left=xs[0]??0,right=xs.at(-1)??1;
 const isValue=numeric;
 let lo=Infinity,hi=-Infinity;
 for(const mode of modes) for(const y of series[mode]) if(isValue(y)){lo=Math.min(lo,y);hi=Math.max(hi,y)}
 const hasValues=Number.isFinite(lo);
 if(!hasValues){lo=0;hi=1}
 if(hasValues&&isValue(opts.threshold)){lo=Math.min(lo,opts.threshold);hi=Math.max(hi,opts.threshold)}
 if(opts.ylim){[lo,hi]=opts.ylim}else{const pad=(hi-lo||Math.max(.01,Math.abs(lo)*.06))*.09;lo-=pad;hi+=pad}
 const sx=x=>left===right?L+PW/2:L+(x-left)/(right-left)*PW;
 const sy=y=>T+(hi-y)/(hi-lo)*PH;
 const txt=(x,y,str,size=12,color='#506478',align='start',bold=false)=>'<text x="'+x+'" y="'+y+'" text-anchor="'+align+'" fill="'+color+'" font-size="'+size+'"'+(bold?' font-weight="650"':'')+'>'+esc(str)+'</text>';
 const path=(d,stroke,width,dash='')=>'<path d="'+d+'" stroke="'+stroke+'" stroke-width="'+width+'" fill="none"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
 const markerId='clip-'+id;
 let svg='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(title)+'" style="background:#fff;font-family:system-ui, sans-serif"><title>'+esc(title)+'</title><rect width="100%" height="100%" fill="white"/><defs><clipPath id="'+markerId+'"><rect x="'+L+'" y="'+T+'" width="'+PW+'" height="'+PH+'"/></clipPath></defs>';
 svg+=txt(3,19,title,W<500?15:17,'#172d43','start',true)+txt(3,43,opts.unit||'');
 let pos=W-154;
 for(const m of modes){svg+=path('M'+pos+',38h18',colors[m],2.5,m==='best'?'6 3':'')+txt(pos+22,42,names[m],12,colors[m]);pos+=73}
 for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,y=sy(v);
  svg+=path('M'+L+','+y+'H'+(W-R),'#dfe6ec',1)+txt(L-7,y+4,nf(v,4),11,'#506478','end');}
 const tickCount=W<540?4:6,lastTicks=new Set();
 for(let j=0;j<tickCount;j++){const x=Math.round(left+(right-left)*j/(tickCount-1));
  if(lastTicks.has(x))continue;lastTicks.add(x);
  svg+=txt(Math.max(L-3,Math.min(W-22,sx(x))),H-26,nf(x,0),11,'#506478','middle');}
 svg+=txt(L+PW/2,H-6,opts.xlabel||'실제 라운드',12,'#506478','middle');
 svg+='<g clip-path="url(#'+markerId+')">';
 if(lo<0&&hi>0)svg+=path('M'+L+','+sy(0)+'H'+(W-R),'#bec9d3',1);
 if(isValue(opts.threshold)&&hasValues){
  svg+=path('M'+L+','+sy(opts.threshold)+'H'+(W-R),'#576979',1.2,'3 4');}
 if(opts.zero&&left<=0&&right>=0)svg+=path('M'+sx(0)+','+T+'V'+(H-B),'#6c7885',1,'3 4');
 for(const m of modes){
  const ys=series[m],step=Math.max(1,Math.ceil(xs.length/PW));
  let d='',open=false,only=null,nvalid=0;
  for(let a=0;a<xs.length;a+=step){
   const end=Math.min(xs.length,a+step);let min=-1,max=-1;const indices=new Set([a,end-1]);
   for(let i=a;i<end;i++){
    if(!isValue(ys[i])){if(i>0&&isValue(ys[i-1]))indices.add(i);continue}
    nvalid++;only=i;
    if(i>0&&!isValue(ys[i-1]))indices.add(i);
    if(min<0||ys[i]<ys[min])min=i;if(max<0||ys[i]>ys[max])max=i;
   }
   if(min>=0){indices.add(min);indices.add(max)}
   for(const i of Array.from(indices).sort((a,b)=>a-b)){
    if(!isValue(ys[i])){open=false;continue}
    d+=(open?'L':'M')+sx(xs[i]).toFixed(2)+','+sy(ys[i]).toFixed(2);open=true;
   }
  }
  svg+=path(d,colors[m],m==='random'?2.25:1.8,m==='best'?'6 3':'');
  if(nvalid===1)svg+='<circle cx="'+sx(xs[only])+'" cy="'+sy(ys[only])+'" r="3.5" fill="'+colors[m]+'"/>';
 }
 svg+='</g>';
 for(const marker of opts.markers||[]){
  if(marker.x<left||marker.x>right)continue;
  const px=sx(marker.x);
  svg+=path('M'+px+','+T+'V'+(H-B),colors[marker.mode],1,'3 5')+
       txt(Math.max(L+2,Math.min(W-33,px+3)),T+12+marker.row*14,marker.name,11,colors[marker.mode],'start',true);
 }
 if(!hasValues)svg+=txt(L+PW/2,T+PH/2,'계산에 필요한 관측이 아직 없어',13,'#596d81','middle');
 svg+='</svg>';
 const hint=opts.hint||'그래프 위에 마우스를 올리면 두 값을 함께 볼 수 있어.';
 host.innerHTML=svg+'<div class="plot-bottom"><span class="hover">'+esc(hint)+'</span><span><button class="svg-save" type="button">SVG</button> <button class="png-save" type="button">PNG</button></span></div>';
 const element=host.querySelector('svg'),hover=host.querySelector('.hover');
 function hoverAt(e){if(!xs.length)return;
  const box=element.getBoundingClientRect(),px=(e.clientX-box.left)*W/box.width;
  const target=left+(px-L)/PW*(right-left);let a=0,b=xs.length-1;
  while(a<b){const mid=Math.floor((a+b)/2);if(xs[mid]<target)a=mid+1;else b=mid}
  const i=a>0&&Math.abs(xs[a-1]-target)<Math.abs(xs[a]-target)?a-1:a;
  hover.textContent=(opts.hoverX||'가로축')+' '+nf(xs[i],0)+(opts.origin?' · 실제 '+nf(xs[i]+opts.origin,0)+'회':'')+'  |  '+modes.map(m=>names[m]+' '+nf(series[m][i],6)).join('   ·   ');
 }
 element.onpointermove=hoverAt;element.onpointerdown=hoverAt;
 const imageBlob=()=>new Blob([new XMLSerializer().serializeToString(element)],{type:'image/svg+xml;charset=utf-8'});
 const filename=id+'-run'+(model.run||1)+'-'+$('phase').value+'-'+$('kind').value;
 host.querySelector('.svg-save').onclick=()=>saveBlob(imageBlob(),filename+'.svg');
 host.querySelector('.png-save').onclick=()=>{
  const url=URL.createObjectURL(imageBlob()),img=new Image();
  img.onload=()=>{const cv=document.createElement('canvas');cv.width=W*2;cv.height=H*2;
   cv.getContext('2d').drawImage(img,0,0,cv.width,cv.height);URL.revokeObjectURL(url);
   cv.toBlob(b=>{if(b)saveBlob(b,filename+'.png')},'image/png')};
  img.onerror=()=>URL.revokeObjectURL(url);img.src=url;
 };
}

"""

TIMELINE = r"""
function timeline(){
 const el=$('timeline'),[start,end]=bounds(),W=Math.max(370,Math.min(1060,el.clientWidth||900)),H=232,L=W<550?95:127,R=27,T=40,last=H-39;
 const records=modes.flatMap(m=>Object.entries(firsts(m)).map(([key,event])=>({m,key,event})));
 let lo=start,hi=end;
 if($('timelineWindow').value==='events'){
  const times=records.filter(x=>x.event).map(x=>x.event.confirmed);
  if(times.length){const first=Math.min(...records.filter(x=>x.event).map(x=>x.event.t)),final=Math.max(...times),pad=Math.max(10,Math.ceil(.06*(final-first+1)));lo=Math.max(start,first-pad);hi=Math.min(end,final+pad)}
 }
 if(hi<lo)hi=lo;
 const sx=t=>L+(t-lo)/(hi-lo||1)*(W-L-R);
 const text=(x,y,v,anchor='start',color='#516579',size=11)=>'<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" fill="'+color+'" font-size="'+size+'">'+esc(v)+'</text>';
 let svg='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" style="font-family:system-ui,sans-serif;background:#fff" role="img" aria-label="두 방식의 첫 사건 시점"><title>두 방식의 첫 사건 시작 라운드</title><rect width="100%" height="100%" fill="white"/>'+text(3,19,'첫 사건의 시작 라운드','start','#172d43',16);
 const eventNames={H:'H↓ 감소',L:'L 패배',V:'V↑ 증폭'};
 const ticks=new Set(interval(0,W<550?3:5).map(i=>Math.round(lo+i/(W<550?3:5)*(hi-lo))));
 for(const tick of ticks){const x=sx(tick);svg+='<path d="M'+x+','+(T-8)+'V'+last+'" stroke="#e0e7ed"/>'+text(x,H-20,nf(tick,0),'middle')}
 records.forEach(({m,key,event},i)=>{
  const y=T+i*25+(i>=3?9:0);
  svg+=text(3,y+4,names[m]+' '+eventNames[key],'start',colors[m],12)+'<path d="M'+L+','+y+'H'+(W-R)+'" stroke="#dce4ec" stroke-dasharray="2 4"/>';
  if(!event){svg+=text(L+7,y-3,'미관측');return}
  const x=sx(event.t),clr=colors[m],right=x>W-91,delta=right?-8:8;
  svg+='<g><title>'+names[m]+' '+letters[key]+' · 시작 '+nf(event.t,0)+'회 · 확인 '+nf(event.confirmed,0)+'회</title>';
  if(key==='H')svg+='<circle cx="'+x+'" cy="'+y+'" r="4.5" fill="'+clr+'"/>';
  else if(key==='L')svg+='<rect x="'+(x-4)+'" y="'+(y-4)+'" width="8" height="8" fill="'+clr+'"/>';
  else svg+='<path d="M'+x+','+(y-5)+'l5,9h-10z" fill="'+clr+'"/>';
  svg+=text(x+delta,y-4,nf(event.t,0),right?'end':'start',clr,12)+'</g>';
 });
 svg+=text((L+W-R)/2,H-3,'실제 라운드','middle')+'</svg>';
 el.innerHTML=svg+'<div class="plot-bottom"><span class="hover">모양은 사건 종류를, 색은 전략 교체 방식을 나타내.</span><button id="timeline-save">SVG 저장</button></div>';
 $('timeline-save').onclick=()=>saveBlob(new Blob([new XMLSerializer().serializeToString(el.querySelector('svg'))],{type:'image/svg+xml;charset=utf-8'}),'event-timeline-'+model.run+'-'+$('phase').value+'.svg');
}

"""

UI = r"""
const blankSide=()=>({series:{},events:{strategy:[],lineage:[],L:[],V:[]},view:null});
let model=LIVE?{config:BOOT.config,engine:BOOT.engine,run:1,round:0,status:'idle',modes:{random:blankSide(),best:blankSide()},completed:[]}:BOOT;
const eventLabels={H_L:'H↓ 발생 → L 발생',L_V:'L 발생 → V↑ 발생',H_V:'H↓ 발생 → V↑ 발생'};
const pairs={H_L:'엔트로피 감소량 E → 손실 D',L_V:'손실 D → 증폭률 A',H_V:'엔트로피 감소량 E → 증폭률 A'};
const stateText={idle:'시작 대기',running:'진행 중',paused:'일시정지',analyzing:'라운드 완료 · 종합 분석 중',between:'다음 회차 대기',done:'완료',error:'오류',closed:'종료'};
let currentCorr=null,corrJob=null,corrRequest=0,autoRound=0,lastDraw=0,lastSignature='',closed=false,polling=false;
const arr=(m,k)=>model.modes[m].series[k]||[];
const interval=(a,b)=>Array.from({length:Math.max(0,b-a+1)},(_,i)=>a+i);
const delta=(a,b)=>a&&b?b.t-a.t:null;
const fmtMode=m=>'<b class="'+m+'">'+names[m]+'</b>';
const rlabel=x=>x?nf(x,0)+'회':'—';
const config=()=>model.config;

const traceNames=['h_strategy','h_lineage','drop_strategy','drop_lineage','gain','total','slope','volatility','crowding','replacements','mutations','loss_degree','amp_pct'];
const tileRows=4096, tileCache=new Map(), tilePromises=new Map();
const traceStem=(m)=>m+'/run_'+String(model.run).padStart(2,'0');
function sampleStride(span){
 let stride=1;
 while(Math.ceil(span/stride)>tileRows)stride*=8;
 return stride;
}
function lowerBound(array,value){
 let lo=0,hi=array.length;
 while(lo<hi){const mid=Math.floor((lo+hi)/2);if(array[mid]<value)lo=mid+1;else hi=mid}
 return lo;
}
function delayFor(key){
 const c=config();
 return key==='L'?c.loss_block*c.loss_repeats-1:key==='V'?c.vol_hold-1:0;
}
function compactEvents(mode,key,start,end){
 const times=model.modes[mode].events[key]||[], delay=delayFor(key);
 const lo=lowerBound(times,start),hi=Math.max(lo,lowerBound(times,end-delay+1));
 const items=[];
 if(hi>lo){
  const picks=new Set();
  for(let i=lo;i<Math.min(hi,lo+80);i++)picks.add(i);
  for(let i=Math.max(lo,hi-20);i<hi;i++)picks.add(i);
  if(key==='L'){
   const auto=lowerBound(times,start+config().compare_window);
   if(auto>=lo&&auto<hi)picks.add(auto);
   const target=Number($('exact-loss-'+mode)?.value || $('loss-'+mode)?.value);
   const hit=lowerBound(times,target);
   if(hit>=lo&&hit<hi&&times[hit]===target)picks.add(hit);
  }
  for(const i of [...picks].sort((a,b)=>a-b))items.push({t:times[i],confirmed:times[i]+delay,index:i-lo+1});
 }
 items.total=hi-lo;
 return items;
}
globalThis.acceptRoundTile=(url,data)=>{
 const bin=atob(data), bytes=new Uint8Array(bin.length);
 for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
 tileCache.set(url,new DataView(bytes.buffer));
 while(tileCache.size>16)tileCache.delete(tileCache.keys().next().value);
};
function openTile(path){
 const url=new URL(path,document.baseURI).href;
 if(tileCache.has(url))return Promise.resolve(tileCache.get(url));
 if(tilePromises.has(url))return tilePromises.get(url);
 const task=new Promise((resolve,reject)=>{
  const el=document.createElement('script');
  el.src=url;
  el.onload=()=>{el.remove();const data=tileCache.get(url);data?resolve(data):reject(Error('그래프 자료를 읽을 수 없어.'))};
  el.onerror=()=>{el.remove();reject(Error('결과 폴더의 그래프 타일을 함께 열어 줘.'))};
  document.head.appendChild(el);
 }).finally(()=>tilePromises.delete(url));
 tilePromises.set(url,task);return task;
}
async function sampledRows(mode,lo,hi,stride,run){
 const first=Math.ceil((lo-1)/stride),last=Math.floor((hi-1)/stride);
 const width=8*(traceNames.length+config().q);
 const result={first:1+first*stride,stride,rows:[]};
 if(first>last)return result;
 const add=(view,i)=>{
  const row=[];
  for(let j=0;j<width;j+=8){
   const x=view.getFloat64(i*width+j,true);row.push(Number.isFinite(x)?x:null);
  }
  result.rows.push(row);
 };
 if(LIVE){
  const url='/api/samples?'+new URLSearchParams({mode,run,lo,hi,stride});
  const res=await fetch(url,{cache:'no-store'});
  if(!res.ok)throw Error((await res.json()).error||'그래프 자료를 불러올 수 없어.');
  const view=new DataView(await res.arrayBuffer());
  if(view.byteLength!==(last-first+1)*width)throw Error('그래프 자료의 길이가 맞지 않아.');
  for(let i=0;i<=last-first;i++)add(view,i);
 }else{
  for(let p=Math.floor(first/tileRows);p<=Math.floor(last/tileRows);p++){
   const file=traceStem(mode)+'_tiles/'+stride+'_'+p+'.js';
   const view=await openTile(file);
   const a=Math.max(first,p*tileRows),b=Math.min(last,(p+1)*tileRows-1);
   for(let i=a;i<=b;i++)add(view,i-p*tileRows);
  }
 }
 return result;
}
async function sampledForPlot(mode,lo,hi,stride,run){
 const side=await sampledRows(mode,lo,hi,stride,run);
 const rounds=side.rows.map((_,i)=>side.first+i*stride);
 if(rounds[0]!==lo){
  const head=await sampledRows(mode,lo,lo,1,run);rounds.unshift(lo);side.rows.unshift(head.rows[0]);
 }
 if(rounds.at(-1)!==hi){
  const tail=await sampledRows(mode,hi,hi,1,run);rounds.push(hi);side.rows.push(tail.rows[0]);
 }
 return {rounds,rows:side.rows};
}
let timeJob=0,afterJob=0;
function appendRawLinks(){
 const el=$('raw-links');
 if(!el)return;
 const large=Boolean(model.large);
 el.hidden=!large||LIVE;
 if(!el.hidden&&!el.hasChildNodes())for(const m of modes){
  const a=document.createElement('a');a.href=traceStem(m)+'_trace.csv';a.download='';a.textContent=names[m]+' 전체 CSV';el.appendChild(a);
 }
 for(const m of modes)$('exact-wrap-'+m).hidden=!large;
}
async function drawLongAfter(){
 const job=++afterJob,run=model.run,c=config(),w=c.compare_window,[lo,hi]=bounds();
 const xs=interval(-w,w-1),series={},anchors=[],table=[],vcol=traceNames.indexOf('volatility');
 try{
  const sides=await Promise.all(modes.map(async m=>{
   const loss=events(m).L,select=$('loss-'+m),chosen=Number($('exact-loss-'+m).value);
   const options='<option value="auto">자동 선택</option>'+loss.map(e=>'<option value="'+e.t+'">'+nf(e.index,0)+'번째 · '+nf(e.t,0)+'회</option>').join('');
   if(select._options!==options){
    const old=select.value;select.innerHTML=options;select._options=options;
    if([...select.options].some(x=>x.value===old))select.value=old;
   }
   const picked=chosen||Number(select.value);
   const anchor=picked?loss.find(e=>e.t===picked):loss.find(e=>e.t-w>=lo&&e.t+w-1<=hi)||loss[0];
   const t=anchor?.t,a=Math.max(lo,(t||lo)-w),b=Math.min(hi,(t||lo)+w-1);
   const data=t?await sampledRows(m,a,b,1,run):{rows:[]};
   const counts=data.rows.map(row=>row.slice(traceNames.length));
   const pre=t&&t-w>=lo?windowRms(counts,t-w-a+1,w,c.n,c.q):null;
   const post=t&&t+w-1<=hi?windowRms(counts,t-a+1,w,c.n,c.q):null;
   const aligned=xs.map(k=>{const v=t?data.rows[t+k-a]?.[vcol]:null;return numeric(v)?100*v:null});
   return {m,t,pre,post,aligned};
  }));
  if(job!==afterJob||run!==model.run)return;
  for(const {m,t,pre,post,aligned} of sides){
   series[m]=aligned;anchors.push(names[m]+' '+rlabel(t));
   table.push('<tr><td>'+fmtMode(m)+'</td><td>'+rlabel(t)+'</td><td>'+nf(numeric(pre)?100*pre:null,4)+'</td><td>'+nf(numeric(post)?100*post:null,4)+'</td><td>'+(pre>0&&numeric(post)?nf(post/pre,4)+'배':'—')+'</td></tr>');
  }
  $('changeRows').innerHTML=table.join('');
  $('afterHelp').textContent='패배 직전과 시작부터 각각 '+nf(w,0)+'라운드의 실제 인원 비율로 계산해. 특정 패배는 시작 라운드를 입력해 선택할 수 있어.';
  chart('aligned','각 방식의 패배를 0에 맞춘 변동 크기',xs,series,{unit:'최근 '+c.vol_window+'회 V (%p)',xlabel:'각 방식의 집단 패배로부터 지난 라운드',zero:true,hint:anchors.join(' · ')});
 }catch(e){if(job===afterJob)$('afterHelp').textContent=e.message}
}
async function downloadSelected(){
 const [lo,hi]=selectedRange(),run=model.run,kind=$('kind').value;
 if(hi<lo)return;
 if(LIVE){
  const a=document.createElement('a');a.href='/api/trace.csv?'+new URLSearchParams({run,lo,hi,kind});a.download='paired-trace-'+run+'.csv';a.click();return;
 }
 const keys=['h_'+kind,'drop_'+kind,'loss_degree','amp_pct','volatility','slope','total'];
 const header=['round',...modes.flatMap(m=>keys.map(k=>m+'_'+k))].join(',')+'\r\n';
 let sink;
 try{
  if(window.showSaveFilePicker){
   const handle=await window.showSaveFilePicker({suggestedName:'paired-trace-'+run+'.csv'});
   sink=await handle.createWritable();
  }else if(hi-lo>=200000){
   $('axisNote').textContent='전체 원자료는 아래 랜덤·베스트 CSV 링크에서 바로 저장할 수 있어. 구간을 줄이면 두 방식을 한 CSV로 저장해.';return;
  }
  const pieces=sink?null:['\ufeff'+header];if(sink)await sink.write('\ufeff'+header);
  for(let a=lo;a<=hi;a+=tileRows){
   const b=Math.min(hi,a+tileRows-1),sides=await Promise.all(modes.map(m=>sampledRows(m,a,b,1,run)));
   const lines=[];
   for(let i=a;i<=b;i++)lines.push([i,...sides.flatMap(s=>keys.map(k=>s.rows[i-a][traceNames.indexOf(k)]??''))].join(','));
   const chunk=lines.join('\r\n')+'\r\n';if(sink)await sink.write(chunk);else pieces.push(chunk);
  }
  if(sink)await sink.close();else saveBlob(new Blob(pieces,{type:'text/csv;charset=utf-8'}),'paired-trace-'+run+'.csv');
 }catch(e){if(sink)await sink.abort().catch(()=>{});if(e.name!=='AbortError')$('axisNote').textContent=e.message}
}

function bounds(){
 const n=model.round,b=config().burn,phase=$('phase').value;
 return phase==='early'?[1,Math.min(b,n)]:phase==='late'?[b+1,n]:[1,n];
}
function events(m){
 const [a,b]=bounds(),e=model.modes[m].events,select=key=>model.large?compactEvents(m,key,a,b):(e[key]||[]).filter(v=>v.t>=a&&v.confirmed<=b);
 return {H:select($('kind').value),L:select('L'),V:select('V')};
}
function firsts(m){const e=events(m);return {H:e.H[0],L:e.L[0],V:e.V[0]}}
function drawTiming(){
 timeline();
 const [a,b]=bounds(),c=config();
 $('scope').textContent=b>=a?'실제 '+nf(a,0)+'–'+nf(b,0)+'라운드':'선택한 구간의 관측 대기';
 $('eventRows').innerHTML=modes.map(m=>{
  const e=events(m),h=e.H[0],l=e.L[0],v=e.V[0];
  const chain=!h||!l||!v?'판정 보류':h.t<l.t&&l.t<v.t&&v.t-h.t<=c.chain_horizon?'조건 충족':'조건 미충족';
  const cell=list=>list.length?'<b>'+nf(list[0].t,0)+'</b><small>확인 '+nf(list[0].confirmed,0)+' · 총 '+nf(list.total??list.length,0)+'건</small>':'미관측';
  return '<tr><td>'+fmtMode(m)+'</td>'+[e.H,e.L,e.V].map(list=>'<td>'+cell(list)+'</td>').join('')+
   '<td>'+signed(delta(h,l))+'</td><td>'+signed(delta(l,v))+'</td><td>'+chain+'</td></tr>';
 }).join('');
 $('firstNote').textContent='각 방식의 첫 H↓ → 첫 L → 첫 V↑ 순서와 '+nf(c.chain_horizon,0)+'라운드 이내 조건을 검사해. 시작과 확인이 모두 선택 구간 안인 사건을 세며, 이동 기준은 직전 구간의 실제 관측을 이어서 사용해.';
}
function selectedRange(){
 const [a,b]=bounds(),w=$('range').value;if(a>b)return [a,b,''];
 if(w==='events'){
  const anchors=modes.flatMap(m=>Object.values(firsts(m)).filter(Boolean).map(e=>e.t));
  if(!anchors.length)return [a,b,'확인된 첫 사건이 생기면 그 주변을 확대해. '];
  const pad=config().compare_window;
  return [Math.max(a,Math.min(...anchors)-pad),Math.min(b,Math.max(...anchors)+pad),''];
 }
 if(w==='custom'){
  const lo=Number($('from').value),hi=Number($('to').value);
  if(!Number.isInteger(lo)||!Number.isInteger(hi)||lo<1||hi<lo)return [a,b,'시작과 끝 라운드를 확인해 줘. '];
  return [Math.max(a,lo),Math.min(b,hi),''];
 }
 return w==='all'?[a,b,'']:[Math.max(a,b-Number(w)+1),b,''];
}
async function drawTime(){
 const job=++timeJob,run=model.run;
 const [a,b,notice]=selectedRange(),c=config(),kind=$('kind').value,aKey=$('anchor').value;
 let origin=0,msg=notice,axis='실제 라운드';
 if(aKey!=='absolute'){
  const [m,k]=aKey.split(':'),ev=firsts(m)[k];
  if(ev){origin=ev.t;axis=names[m]+' '+letters[k]+' 기준 상대 라운드';msg+='두 선의 0은 실제 '+nf(origin,0)+'라운드야. ';}
  else msg+='선택한 기준 사건이 미관측이라 실제 라운드로 표시해. ';
 }
 $('axisNote').textContent=msg+(b>=a?'범위 '+nf(a,0)+'–'+nf(b,0)+'회. ':'해당 범위의 관측 대기. ')+'세로 점선의 색은 방식, H↓·L·V↑는 그 방식에서 처음 발생한 사건을 뜻해.';
 let sampled=null,xs;
 if(model.large&&b>=a){
  const stride=sampleStride(b-a+1);
  if(stride>1)$('axisNote').textContent+=' 선은 '+nf(stride,0)+'라운드 간격으로 표시해. 사건과 통계는 전체 라운드로 계산해.';
  try{sampled=Object.fromEntries(await Promise.all(modes.map(async m=>[m,await sampledForPlot(m,a,b,stride,run)])))}
  catch(e){if(job===timeJob)$('axisNote').textContent=e.message;return}
  if(job!==timeJob||run!==model.run)return;
  xs=sampled.random.rounds.map(x=>x-origin);
 }else xs=interval(a,b).map(x=>x-origin);
 const marks=[];
 for(const [mi,m] of modes.entries())for(const [j,k] of ['H','L','V'].entries()){
  const ev=firsts(m)[k];if(ev)marks.push({x:ev.t-origin,mode:m,name:letters[k],row:mi*3+j});
 }
 function draw(id,title,key,unit,cut,factor=1){
  const s={};for(const m of modes){const vals=sampled?sampled[m].rows.map(row=>row[traceNames.indexOf(key)]):arr(m,key).slice(a-1,b);s[m]=factor===1?vals:vals.map(x=>numeric(x)?x*factor:null)}
  chart(id,title,xs,s,{unit,threshold:cut,xlabel:axis,hoverX:axis,origin,markers:marks});
 }
 draw('hplot',kind==='strategy'?'전략 분포 엔트로피 H':'계보 분포 엔트로피 H','h_'+kind,'정규화 엔트로피');
 draw('eplot','엔트로피 감소량 E','drop_'+kind,'엔트로피 차이 · 기준 '+nf(c.h_drop,4),c.h_drop);
 draw('dplot','집단 손실 D','loss_degree','점수 / 1인 / 라운드');
 const cutoff=100*(c.vol_ratio-1);
 draw('aplot','진동 증폭률 A','amp_pct','증가율 (%) · 기준 '+nf(cutoff)+'%',cutoff);
 if($('extras').open){
  draw('vplot','선택 인원 비율의 시간 변동 V','volatility','인원 비율 변동 (%p)',null,100);
  draw('splot','집단 점수의 평균 기울기','slope',c.loss_block+'라운드 평균 · 점수 / 라운드');
  draw('tplot','집단의 누적 점수','total','점수');
 }
}
function windowRms(counts,start,w,n,q){
 const sums=Array(q).fill(0),sq=Array(q).fill(0);
 for(let i=start-1;i<start+w-1;i++){
  const row=counts[i];if(!row)return null;
  for(let j=0;j<q;j++){sums[j]+=row[j];sq[j]+=row[j]*row[j]}
 }
 return Math.sqrt(Math.max(0,sq.reduce((acc,s,j)=>acc+w*s-sums[j]*sums[j],0)/q))/(w*n);
}
const lossSignatures={};
function drawAfter(){
 if(model.large){drawLongAfter();return}
 ++afterJob;
 const c=config(),w=c.compare_window,[lo,hi]=bounds(),xs=interval(-w,w-1),series={},anchors=[];
 const rows=modes.map(m=>{
  const loss=events(m).L,select=$('loss-'+m),phase=$('phase').value;
  const sign=[model.run,phase,loss.length,loss[0]?.t,loss.at(-1)?.t].join('/');
  if(sign!==lossSignatures[m]){
   const keep=lossSignatures[m]?.split('/').slice(0,2).join('/')===[model.run,phase].join('/')?select.value:'auto';
   select.innerHTML='<option value="auto">자동 선택</option>'+loss.map((e,i)=>'<option value="'+e.t+'">'+(i+1)+'번째 · '+nf(e.t,0)+'회</option>').join('');
   if(keep==='auto'||loss.some(e=>String(e.t)===keep))select.value=keep;lossSignatures[m]=sign;
  }
  const anchor=select.value==='auto'?(loss.find(e=>e.t-w>=lo&&e.t+w-1<=hi)||loss[0]):loss.find(e=>e.t===Number(select.value));
  const t=anchor?.t,counts=arr(m,'counts');
  const pre=t&&t-w>=lo?windowRms(counts,t-w,w,c.n,c.q):null;
  const post=t&&t+w-1<=hi?windowRms(counts,t,w,c.n,c.q):null;
  const ratio=pre>0&&numeric(post)?post/pre:null;
  const v=arr(m,'volatility');
  series[m]=xs.map(k=>{const i=t+k-1;return t&&t+k>=lo&&t+k<=hi&&numeric(v[i])?100*v[i]:null});
  anchors.push(names[m]+' '+rlabel(t));
  return '<tr><td>'+fmtMode(m)+'</td><td>'+rlabel(t)+'</td><td>'+nf(numeric(pre)?100*pre:null,4)+'</td><td>'+nf(numeric(post)?100*post:null,4)+'</td><td>'+(numeric(ratio)?nf(ratio,4)+'배':'—')+'</td></tr>';
 });
 $('changeRows').innerHTML=rows.join('');
 $('afterHelp').textContent='표는 패배 직전 '+w+'회와 패배 시작부터 '+w+'회의 V를 각각 새로 계산해 (%p). 1배를 넘으면 사후 변동이 더 커. 자동 선택은 두 구간이 모두 있는 첫 패배를 사용하고, 아직 없으면 첫 패배를 보여 줘. 불완전한 구간은 —로 표시해.';
 chart('aligned','각 방식의 패배를 0에 맞춘 변동 크기',xs,series,{unit:'최근 '+c.vol_window+'회 V (%p)',xlabel:'각 방식의 집단 패배로부터 지난 라운드',zero:true,hint:'각 선의 0에 해당하는 실제 라운드  |  '+anchors.join(' · ')});
}
function corrKey(){return [model.run,$('kind').value,$('phase').value,$('lagBasis').value].join('/')}
function corrSnapshot(){
 if(!LIVE){const k=$('kind').value,p=$('phase').value;return {round:model.round,modes:Object.fromEntries(modes.map(m=>[m,BOOT.curves[m][p][k][$('lagBasis').value==='events'?'event_curves':'curves']]))}}
 return currentCorr?.key===corrKey()?currentCorr:null;
}
function drawLag(){
 const result=corrSnapshot();
 $('export-lag').disabled=!result;
 if(!result){
  $('lagRows').innerHTML='';for(const k of Object.keys(pairs))$('lag-'+k).innerHTML='';
  $('lagStatus').textContent=corrJob?'현재 관측값으로 계산 중…':'현재까지 상관 계산을 누르면 세 변수 쌍의 결과가 표시돼.';return;
 }
 $('lagStatus').textContent=nf(result.round,0)+'라운드까지 계산 · 시차 ±'+config().max_lag+'라운드 · 간격 '+config().lag_step+'라운드 · 변화가 없거나 유효 짝이 3개 미만이면 —';
 let rows='';
 for(const [k,label] of Object.entries($('lagBasis').value==='events'?eventLabels:pairs)){
  const curves={},series={};
  for(const m of modes){
   const curve=result.modes[m][k];curves[m]=curve;series[m]=curve.map(p=>p.r);
   const zero=curve.find(r=>r.lag===0),valid=curve.filter(r=>numeric(r.r));
   valid.sort((a,b)=>Math.abs(b.r)-Math.abs(a.r)||Math.abs(a.lag)-Math.abs(b.lag)||a.lag-b.lag);
   const peak=valid[0];
   rows+='<tr><td>'+label+'</td><td>'+fmtMode(m)+'</td><td>'+nf(zero?.r,4)+'</td><td>'+nf(zero?.n,0)+'</td><td>'+signed(peak?.lag)+'</td><td>'+nf(peak?.r,4)+'</td><td>'+nf(peak?.n,0)+'</td></tr>';
  }
  chart('lag-'+k,label,curves.random.map(p=>p.lag),series,{unit:'피어슨 r',xlabel:'시차 k (라운드) · 양수이면 앞 변수가 선행',ylim:[-1,1],zero:true,hoverX:'시차 k'});
 }
 $('lagRows').innerHTML=rows;
}
async function calculate(){
 const key=corrKey(),num=++corrRequest;
 corrJob=key;drawLag();if($('calc'))$('calc').disabled=true;
 try{
  const url='/api/correlation?run='+model.run+'&kind='+$('kind').value+'&phase='+$('phase').value+'&basis='+$('lagBasis').value;
  const res=await fetch(url,{cache:'no-store'}),d=await res.json();if(!res.ok)throw Error(d.error||'상관 계산 실패');
  if(key===corrKey()&&num===corrRequest)currentCorr={...d,key};
 }catch(e){if(num===corrRequest&&key===corrKey())$('error').textContent=e.message}
 finally{if(num===corrRequest){corrJob=null;if($('calc'))$('calc').disabled=false;drawLag()}}
}
function exportCsv(filename,rows){
 const contents=Array.from(rows,row=>row.map(v=>v==null?'':String(v)).join(',')).join('\r\n');
 saveBlob(new Blob(['\ufeff',contents],{type:'text/csv;charset=utf-8'}),filename);
}
$('export-events').onclick=()=>{
 const rows=[['run','phase','mode','event','start_round','confirmed_round']];
 for(const m of modes){
  if(model.large){
   const [a,b]=bounds();
   for(const key of [$('kind').value,'L','V']){
    const times=model.modes[m].events[key]||[],delay=delayFor(key);
    const end=lowerBound(times,b-delay+1);
    for(let i=lowerBound(times,a);i<end;i++)rows.push([model.run,$('phase').value,m,key===$('kind').value?'H_'+key:key,times[i],times[i]+delay]);
   }
  }else for(const [k,group] of Object.entries(events(m)))for(const e of group)rows.push([model.run,$('phase').value,m,k==='H'?'H_'+$('kind').value:k,e.t,e.confirmed]);
 }
 exportCsv('events-'+model.run+'.csv',rows);
};
$('export-trace').onclick=()=>{
 if(model.large){downloadSelected();return}
 const [a,b]=selectedRange(),kind=$('kind').value,keys=['h_'+kind,'drop_'+kind,'loss_degree','amp_pct','volatility','slope','total'];
 const rows=[['round',...modes.flatMap(m=>keys.map(k=>m+'_'+k))]];
 for(let t=a;t<=b;t++)rows.push([t,...modes.flatMap(m=>keys.map(k=>arr(m,k)[t-1]))]);
 exportCsv('paired-trace-'+model.run+'.csv',rows);
};
$('export-lag').onclick=()=>{
 const d=corrSnapshot();if(!d)return;
 const rows=[['run','through_round','phase','entropy','basis','pair','mode','lag','r','n_pairs']];
 for(const k of Object.keys(pairs))for(const m of modes)for(const v of d.modes[m][k])rows.push([model.run,d.round,$('phase').value,$('kind').value,$('lagBasis').value,k,m,v.lag,v.r,v.n]);
 exportCsv('lag-'+model.run+'.csv',rows);
};
function redraw(full=true){
 drawTiming();drawTime();drawAfter();if(full)drawLag();lastDraw=Date.now();
}
$('timelineWindow').onchange=timeline;
$('lagBasis').onchange=()=>{drawLag();if(LIVE&&model.round>0)calculate()};
for(const id of ['phase','kind'])$(id).onchange=()=>{redraw();if(LIVE&&model.round>0)calculate()};
$('range').onchange=()=>{$('custom').hidden=$('range').value!=='custom';drawTime()};
$('anchor').onchange=drawTime;$('applyRange').onclick=drawTime;$('extras').ontoggle=drawTime;
for(const m of modes){
 $('loss-'+m).onchange=()=>{$('exact-loss-'+m).value='';drawAfter()};
 $('exact-loss-'+m).onchange=drawAfter;
}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{redraw();if(LIVE)drawWorlds()},130)});
const boardState={random:{points:[],selected:null},best:{points:[],selected:null}};
function drawWorlds(){
 for(const m of modes){
  const canvas=$('world-'+m),box=canvas.getBoundingClientRect(),W=Math.max(240,box.width),H=box.height||168;
  const scale=Math.min(2,window.devicePixelRatio||1),ctx=canvas.getContext('2d');
  if(!ctx)continue;
  canvas.width=Math.round(W*scale);canvas.height=Math.round(H*scale);ctx.setTransform(scale,0,0,scale,0,0);
  const q=config().q,b=boardState[m],view=model.modes[m].view;
  const cols=Math.min(4,q),bands=Math.ceil(q/cols),w=W/cols,h=H/bands;
  const counts=view?(view.counts||arr(m,'counts').at(-1)):Array(q).fill(0),slots=Array(q).fill(0),target=[];
  if(view)for(const action of view.actions){
   const i=slots[action]++,nc=Math.max(1,Math.ceil(Math.sqrt(counts[action]*w/Math.max(12,h-24)))),nr=Math.max(1,Math.ceil(counts[action]/nc));
   target.push({x:(action%cols)*w+5+(i%nc+.5)*(w-10)/nc,y:Math.floor(action/cols)*h+23+(Math.floor(i/nc)+.5)*(h-28)/nr});
  }
  const replaced=new Set(view?.replaced||[]),start=b.points.length===target.length?b.points:target;
  const since=performance.now(),r=Math.max(1.1,Math.min(4.5,Math.sqrt(W*H/Math.max(20,config().n))*.22));
  cancelAnimationFrame(b.animation);
  const paint=progress=>{
   ctx.clearRect(0,0,W,H);ctx.font='12px system-ui, sans-serif';ctx.textAlign='center';
   for(let j=0;j<q;j++){
    const x=j%cols*w,y=Math.floor(j/cols)*h;ctx.fillStyle=j===view?.winner?'#e9f0f5':'#f5f7f9';ctx.fillRect(x+1,y+1,w-2,h-2);
    ctx.fillStyle='#364c62';ctx.fillText('선택 '+(j+1)+' · '+(counts?.[j]??0)+'명',x+w/2,y+15);
   }
   b.points=target.map((p,i)=>({x:start[i].x+(p.x-start[i].x)*progress,y:start[i].y+(p.y-start[i].y)*progress}));
   if(!view)return;
   for(const [child,parent] of view.parents.slice(0,30)){const a=b.points[child],z=b.points[parent];ctx.strokeStyle=colors[m];ctx.globalAlpha=.45;ctx.beginPath();ctx.moveTo(z.x,z.y);ctx.lineTo(a.x,a.y);ctx.stroke();ctx.globalAlpha=1}
   for(const [i,p] of b.points.entries()){
    ctx.fillStyle=view.actions[i]===view.winner?colors[m]:(m==='random'?'#95bce4':'#deb28c');
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,2*Math.PI);ctx.fill();
    if(replaced.has(i)||b.selected===i){ctx.strokeStyle='#25394c';ctx.lineWidth=b.selected===i?2:1;ctx.stroke();ctx.lineWidth=1}
   }
  };
  function animate(now){const pct=Math.min(1,(now-since)/180);paint(pct);if(pct<1)b.animation=requestAnimationFrame(animate)}
  paint(0);b.animation=requestAnimationFrame(animate);
  function describe(){const i=b.selected;if(!view||i==null||i>=view.actions.length)return;
   $('agent-'+m).textContent=(i+1)+'번 · 선택 '+(view.actions[i]+1)+' · 확률 ('+view.probabilities[i].map(x=>nf(x,3)).join(', ')+') · 점수 '+nf(view.scores[i])+' · 계보 '+view.lineages[i]+(replaced.has(i)?' · 전략 교체':'');
  }
  describe();canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let nearest=null,d=144;
   b.points.forEach((p,i)=>{const dd=(p.x-x)**2+(p.y-y)**2;if(dd<d){d=dd;nearest=i}});
   if(nearest!==null){b.selected=nearest;describe();cancelAnimationFrame(b.animation);paint(1)}
  };
 }
}
function header(){
 const c=config();
 $('meta').textContent=nf(c.n,0)+'명 · 선택지 '+c.q+'개 · 회차당 '+nf(c.rounds,0)+'라운드 · 베스트 변이 확률 '+nf(100*c.mutation_prob,2)+'% · 시드 '+c.seed+' · '+model.engine;
 $('clock').textContent=model.run+'회차  /  '+nf(model.round,0)+' / '+nf(c.rounds,0)+' 라운드'+(LIVE?' · '+(stateText[model.status]||''):' · 완료');
}
let displayedSpeed = 0;
function syncSpeedControl(){
 const rate = model.speed ?? 0;
 if(rate === displayedSpeed)return;
 const picker = $('speed'), form = $('customSpeedForm'), input = $('customSpeed');
 const editing = !form.hidden && (input === document.activeElement || input.value !== String(displayedSpeed || ''));
 displayedSpeed = rate;
 $('actualSpeed').textContent = rate ? '설정 '+nf(rate,4)+' 라운드/초' : '설정 최고속';
 if(editing)return;
 const value = rate ? String(rate) : 'max';
 const preset = Array.from(picker.options).some(option => option.value === value);
 picker.value = preset ? value : 'custom';
 form.hidden = preset;
 if(!preset)input.value = value;
}
function liveBar(){
 const s=model.status,c=config(),busy=['analyzing','done','error','closed'].includes(s);
 $('start').disabled=busy||s==='running';$('start').textContent=s==='paused'?'두 방식 계속':s==='between'?'다음 회차 시작':'두 방식 시작';
 $('step').disabled=busy;$('pause').disabled=s!=='running';$('finish').disabled=busy;
 $('measure').disabled=busy||(model.round>=c.burn&&s!=='between')||!c.burn;
 syncSpeedControl();
 $('measuredSpeed').textContent='실측 '+(s==='running'?nf(model.measured_speed??0,1):'0')+' 라운드/초';
 $('settingsFields').disabled=s!=='idle';$('auto').checked=model.auto_next??true;
 $('bar').style.width=(100*model.round/c.rounds)+'%';
 const archive=model.archive;
 if(archive){
  $('archiveStart').disabled=archive.busy;
  const status=$('archiveState'),link=$('archiveLink');
  if(status.textContent!==archive.message)status.textContent=archive.message;
  link.hidden=!archive.url;
  if(archive.url)link.setAttribute('href',archive.url);
  else link.removeAttribute('href');
 }
 const completed=model.completed||[];
 $('reports').innerHTML=completed.length?completed.map(r=>'<a class="pill" href="/files/compare_'+String(r).padStart(2,'0')+'.html" target="_blank" rel="noopener">'+r+'회차 비교 ↗</a>').join(' ')+' <a href="/files/index.html" target="_blank" rel="noopener">반복 요약 ↗</a>':'';
 if(model.error)$('error').textContent=model.error;
}
async function control(action,extra={}){
 try{
  const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Experiment-Token':BOOT.token},body:JSON.stringify({action,...extra})});
  const result=await response.json();if(!response.ok)throw Error(result.error||'조작 실패');
  if(action==='speed'&&result.speed_revision>=(model.speed_revision??0)){
   model.speed_revision=result.speed_revision;model.speed=result.speed;syncSpeedControl();
  }
  if(action==='settings')$('settingsAck').textContent='회차당 '+nf(result.rounds,0)+'라운드 적용됨';
  $('error').textContent='';
  if(action==='stop'){closed=true;model.status='closed';header();liveBar();return}
  await refresh();
 }catch(e){$('error').textContent=e.message}
}
async function refresh(){
 if(polling||closed)return;polling=true;
 try{
  const seen=$('completion').dataset.completed||0;
  const res=await fetch('/api/state?run='+model.run+'&since='+model.round+'&seen='+seen,{cache:'no-store'});
  if(!res.ok)throw Error('실험 상태를 불러올 수 없어.');const next=await res.json();
  const changed=next.run!==model.run||next.round!==model.round,reset=next.start===0;
  const cfgChanged=JSON.stringify(model.config)!==JSON.stringify(next.config);
  const differentRun=next.run!==model.run;
  for(const m of modes){
   const inc=next.modes[m],side=model.modes[m];
   if(next.large){
    if(reset||!model.large)side.events=inc.events;
    else for(const [k,values] of Object.entries(inc.events))for(const t of values)side.events[k].push(t);
    side.series={};
   }else{
    if(reset)side.series=inc.series;else for(const [k,values] of Object.entries(inc.series)){for(const x of values)side.series[k].push(x)}
    side.events=inc.events;
   }
   side.view=inc.view;
  }
  if(differentRun){autoRound=0;currentCorr=null;for(const m of modes)boardState[m].selected=null}
  if(next.speed_revision<(model.speed_revision??0)){next.speed_revision=model.speed_revision;next.speed=model.speed}
  const sides=model.modes;model={...next,modes:sides};
  if(next.definitions&&cfgChanged)$('methodHost').innerHTML=next.definitions;
  if(cfgChanged)for(const input of $('fields').querySelectorAll('input'))input.value=model.config[input.name];
  appendRawLinks();receiveCompletedResult(next);header();liveBar();const sig=[model.run,model.round,model.status,$('phase').value,$('kind').value].join('/');
  if(sig!==lastSignature||cfgChanged){if(changed||reset)drawWorlds();
   if(Date.now()-lastDraw>1050||model.status!=='running'||differentRun||cfgChanged){redraw(differentRun);lastSignature=sig}
  }
  const done=model.status==='done'||model.status==='between';
  if(!corrJob&&model.round>0&&(!model.large||done||model.round<=100000)&&((done&&corrSnapshot()?.round!==model.round)||(!done&&model.round>=Math.max(500,autoRound*2)))){
   autoRound=model.round;calculate();
  }
 }catch(e){if(!closed)$('error').textContent=e.message}finally{polling=false}
}
if(LIVE){
 const fields={n:'에이전트 수',q:'선택지 수',rounds:'회차당 라운드',repeats:'반복 수',burn:'초기 적응 라운드',seed:'시드',bins:'확률축 구간 수',mutation_prob:'변이 확률 (0~1)',mutation_sd:'변이 표준편차',probe_round:'개입 시작 라운드 (0 자동)',chain_horizon:'후속 관측 라운드'};
 $('fields').innerHTML=Object.entries(fields).map(([k,label])=>'<label>'+label+'<input required type="number" name="'+k+'" value="'+config()[k]+'" step="'+(k.startsWith('mutation')?'any':'1')+'"></label>').join('');
 $('settingsForm').onsubmit=e=>{e.preventDefault();control('settings',{config:Object.fromEntries(new FormData(e.target))})};
 for(const id of ['start','pause','step','finish','measure','stop'])$(id).onclick=async()=>{
  const first=model.status==='idle'&&['start','step','finish','measure'].includes(id);
  const form=$('settingsForm');
  if(first&&!form.reportValidity())return;
  if(first&&picker.value==='custom'&&speedInput.value!==''&&speedInput.valueAsNumber>0&&speedInput.valueAsNumber!==model.speed){
   applySpeed(speedInput.value);
  }
  await speedRequests;
  await control(id,first?{config:Object.fromEntries(new FormData(form))}:{});
 };
 const picker = $('speed'), speedForm = $('customSpeedForm'), speedInput = $('customSpeed');
 for(const input of [speedInput,$('fields').querySelector('[name=rounds]')]){
  input.addEventListener('paste',e=>{
   const value=e.clipboardData.getData('text').trim();
   if(/^\d{1,3}(?:,\d{3})+(?:\.\d+)?$/.test(value)){
    e.preventDefault();input.value=value.replace(/,/g,'');input.dispatchEvent(new Event('input',{bubbles:true}));
   }
  });
 }
 let speedRequests = Promise.resolve(), speedTimer;
 function applySpeed(value){
  clearTimeout(speedTimer);
  speedRequests = speedRequests.then(() => {
   const rate=value==='max'?0:Number(value);
   if(rate!==model.speed)return control('speed',{value});
  });
 }
 picker.onchange = () => {
  clearTimeout(speedTimer);
  speedForm.hidden = picker.value !== 'custom';
  if(speedForm.hidden){applySpeed(picker.value);return}
  speedInput.setCustomValidity('');
  speedInput.value = String(model.speed || '');
  speedInput.focus();
  speedInput.select();
 };
 speedInput.oninput = () => {
  speedInput.setCustomValidity('');
  clearTimeout(speedTimer);
  if(speedInput.valueAsNumber>0 && Number.isFinite(speedInput.valueAsNumber)){
   speedTimer=setTimeout(()=>{if(picker.value==='custom')applySpeed(speedInput.value)},450);
  }
 };
 speedForm.onsubmit = event => {
  event.preventDefault();
  if(!Number.isFinite(speedInput.valueAsNumber) || speedInput.valueAsNumber <= 0){
   speedInput.setCustomValidity('0보다 큰 초당 라운드 수를 입력해 줘.');
   speedInput.reportValidity();
   return;
  }
  applySpeed(speedInput.value);
 };
 $('archiveForm').onsubmit=e=>{e.preventDefault();control('archive',{path:$('archivePath').value})};
 $('auto').onchange=()=>control('auto',{value:$('auto').checked});
 $('calc').onclick=calculate;liveBar();drawWorlds();refresh();setInterval(refresh,650);
}
appendRawLinks();header();redraw();

"""

SUMMARY_UI = r"""
const model={run:'summary'};
function showMeans(){
 const phase=$('phase').value,kind=$('kind').value;
 for(const tr of document.querySelectorAll('tr[data-phase]'))tr.hidden=tr.dataset.phase!==phase||tr.dataset.kind!==kind;
 const basis=$('lagBasis').value, aggregate=GROUP[phase][kind][basis==='events'?'event_curves':'curves'];
 const labels=basis==='events'?{H_L:'H↓ 발생 → L 발생',L_V:'L 발생 → V↑ 발생',H_V:'H↓ 발생 → V↑ 발생'}:{H_L:'엔트로피 감소량 E → 손실 D',L_V:'손실 D → 증폭률 A',H_V:'엔트로피 감소량 E → 증폭률 A'};
 for(const [k,title] of Object.entries(labels)){
  const xs=aggregate.random[k].map(p=>p.lag);
  const vals=Object.fromEntries(modes.map(m=>[m,aggregate[m][k].map(p=>p.r)]));
  const n=modes.map(m=>names[m]+' '+(aggregate[m][k].find(v=>v.lag===0)?.n_runs||0)+'회').join(' · ');
  chart('mean-'+k,title,xs,vals,{unit:'평균 피어슨 r',ylim:[-1,1],zero:true,hoverX:'시차 k',xlabel:'시차 k (라운드) · 양수이면 앞 변수가 선행',hint:'시차 0의 유효 회차  |  '+n});
 }
}
$('phase').onchange=showMeans;$('kind').onchange=showMeans;$('lagBasis').onchange=showMeans;window.addEventListener('resize',showMeans);showMeans();

"""

def shell(title, body, script=''):
    return ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1"><title>'+html.escape(title)+'</title><style>'+
            STYLE+'</style></head><body>'+body+'<script>'+script+'</script></body></html>')


def dashboard(c, data, live, completion=None):
    if not live and data['round'] == c.rounds and completion is None:
        # 저장된 비교 화면에 포함된 자료로도 같은 완료 보고서를 만들 수 있다.
        sides = {}
        for m in MODES:
            source = data['modes'][m]
            windows = window_changes(source['series'],source['events']['L'],c)
            ratios = [post/pre for pre,post in windows.values() if pre and post is not None]
            sides[m] = completed_side(c,source['events'],data['curves'][m],ratios)
        completion = completion_content(c,{data['run']:sides},None)
    stages = ''.join(f'<option value="{k}">{PHASE_NAMES[k]}</option>' for k in PHASES)
    livebar = """
<section id="controls"><div class="actions"><button class="primary" id="start">두 방식 시작</button>
<button id="pause">일시정지</button><button id="step">한 라운드</button>
<label>속도 <select id="speed"><option value="1">1회/초</option><option value="4">4회/초</option>
<option value="25">25회/초</option><option value="150">150회/초</option><option value="2000">2,000회/초</option><option value="100000">100,000회/초</option><option value="max" selected>최고속</option><option value="custom">직접 입력</option></select></label>
<form class="actions" id="customSpeedForm" hidden><label>라운드/초 <input id="customSpeed" type="number" min="0" step="any" required placeholder="예) 100000"></label>
<button type="submit">속도 적용</button></form><output class="helper" id="actualSpeed" aria-live="polite">설정 최고속</output><span class="helper" id="measuredSpeed" title="두 방식이 모두 마친 라운드를 기준으로 측정해.">실측 0 라운드/초</span>
<button id="measure">적응기 끝까지</button><button id="finish">이번 회차 끝까지</button>
<label class="check"><input type="checkbox" id="auto" checked> 다음 회차 자동 진행</label>
<button id="stop">종료</button></div><div class="progress"><div id="bar"></div></div><div id="error" role="alert"></div>
<details id="settings"><summary>실험 설정</summary><form id="settingsForm"><fieldset id="settingsFields"><div class="settings-grid" id="fields"></div><button type="submit">설정 적용</button> <output id="settingsAck" class="helper" aria-live="polite"></output></fieldset></form></details>
<details><summary>기존 결과 폴더 분석</summary><form id="archiveForm" class="actions"><label>결과 폴더 경로 <input id="archivePath" type="text" required placeholder="config.json이 있는 폴더"></label><button id="archiveStart">분석 시작</button>
<output id="archiveState" class="helper" aria-live="polite"></output><a id="archiveLink" hidden target="_blank" rel="noopener">분석 결과 열기</a></form></details></section>
<section><div class="section-head"><h2>에이전트의 선택</h2><span>점 하나 = 한 명</span></div><div class="worlds">"""
    if live:
        for m in MODES:
            livebar += f'<div class="world {m}"><h3>{MODE_NAMES[m]} <small>{LABELS[m]}</small></h3><canvas id="world-{m}" aria-label="{MODE_NAMES[m]} 선택지별 에이전트"></canvas><p id="agent-{m}" class="helper">점을 누르면 해당 에이전트의 전략과 점수를 볼 수 있어.</p></div>'
        livebar += '</div><p class="helper">짙은 점은 승리, 테두리는 전략 교체를 뜻해. 연결선은 전략을 복사한 사람과 원본을 제공한 사람을 이어 줘. 방식별로 최대 30개까지 표시해.</p></section>'
    body = f"""<main><header><div class="eyebrow">2026 영재원 연구 · 진화 소수파 게임</div><h1>전략 교체 방식 비교</h1>
<p id="meta">{c.n}명 · {c.q}개 선택지 · {c.rounds:,}라운드</p><div class="mast"><div class="key"><span class="random">랜덤 · 무작위 참가자 모방</span><span class="best">베스트 · 최고점자 모방</span></div><strong id="clock" aria-live="off"></strong></div></header>
{result_section(completion or '')}{livebar if live else ''}<nav><a href="#timing">사건 시점</a><a href="#times">시계열</a><a href="#after">패배 전후</a><a href="#lags">시차 상관</a><a href="#definitions">판정 기준</a></nav>
<div class="analysis-filter"><label>분석 구간 <select id="phase">{stages}</select></label>
<label>엔트로피 <select id="kind"><option value="strategy">전략 분포</option><option value="lineage">계보 분포</option></select></label><span class="helper" id="scope"></span></div>
<section id="timing"><div class="section-head"><h2>세 사건은 언제 일어났을까?</h2><button id="export-events">사건 CSV</button></div>
<p>각 칸은 <b>첫 시작 라운드</b>와 확인 라운드를 보여 줘. 두 시간 차이가 양수면 왼쪽 사건이 먼저야.</p>
<div class="scroll"><table><thead><tr><th>방식</th><th>H↓ 엔트로피 감소</th><th>L 집단 패배</th><th>V↑ 진동 증폭</th><th>L − H</th><th>V − L</th><th>첫 사건 연쇄</th></tr></thead><tbody id="eventRows"></tbody></table></div>
<p class="helper" id="firstNote"></p><div class="actions"><label>사건 그래프 범위 <select id="timelineWindow"><option value="events">첫 사건 주변 확대</option><option value="all">선택 구간 전체</option></select></label></div><div id="timeline" class="plot"></div></section>
<section id="times"><div class="section-head"><h2>같은 라운드에서 비교</h2><button id="export-trace">시계열 CSV</button></div><div class="actions">
<label>표시 범위 <select id="range"><option value="all">선택 구간 전체</option><option value="events">첫 사건 주변</option><option value="500">최근 500회</option><option value="2000">최근 2,000회</option><option value="custom">직접 입력</option></select></label>
<span id="custom" hidden><label>시작 <input id="from" type="number" value="1" min="1"></label> <label>끝 <input id="to" type="number" value="{c.rounds}" min="1"></label> <button id="applyRange">적용</button></span>
<label>x축 원점 <select id="anchor"><option value="absolute">실제 라운드</option><option value="random:H">랜덤 H↓ = 0</option><option value="best:H">베스트 H↓ = 0</option><option value="random:L">랜덤 L = 0</option><option value="best:L">베스트 L = 0</option><option value="random:V">랜덤 V↑ = 0</option><option value="best:V">베스트 V↑ = 0</option></select></label></div>
<p class="actions" id="raw-links" hidden></p><p class="helper" id="axisNote"></p><div id="hplot" class="plot"></div><div id="eplot" class="plot"></div><div id="dplot" class="plot"></div><div id="aplot" class="plot"></div>
<details id="extras"><summary>변동 크기 V · 평균 기울기 · 누적 점수</summary><div id="vplot" class="plot"></div><div id="splot" class="plot"></div><div id="tplot" class="plot"></div></details></section>
<section id="after"><h2>집단 패배 전후에 변동이 커졌을까?</h2><p id="afterHelp"></p>
<div class="actions"><label>랜덤 기준 사건 <select id="loss-random"></select></label><label>베스트 기준 사건 <select id="loss-best"></select></label><label id="exact-wrap-random" hidden>랜덤 패배 시작 라운드 <input type="number" min="1" id="exact-loss-random"></label><label id="exact-wrap-best" hidden>베스트 패배 시작 라운드 <input type="number" min="1" id="exact-loss-best"></label></div>
<div class="scroll"><table><thead><tr><th>방식</th><th>기준 패배</th><th>전 구간 V</th><th>후 구간 V</th><th>후 / 전</th></tr></thead><tbody id="changeRows"></tbody></table></div>
<div id="aligned" class="plot"></div></section>
<section id="lags"><div class="section-head"><h2>시차 상관</h2><button id="export-lag">상관 CSV</button></div>
<p>앞 변수의 t라운드 값과 뒤 변수의 t+k라운드 값을 짝지어 피어슨 상관계수 r을 구해. k = +10이면 앞 변수가 10라운드 먼저야. r이 양수면 함께 증가하는 경향, 음수면 반대로 움직이는 경향을 뜻해.</p>
<div class="actions"><label>상관 분석 대상 <select id="lagBasis"><option value="continuous">측정값 E · D · A</option><option value="events">사건 발생 여부 (0·1)</option></select></label>{'<button id="calc" class="primary">현재까지 상관 계산</button>' if live else ''}<span id="lagStatus" class="helper"></span></div>
<div class="scroll"><table><thead><tr><th>변수 쌍</th><th>방식</th><th>같은 라운드 r</th><th>유효 짝 수</th><th>최대 |r|의 시차 k</th><th>그 시차의 r</th><th>유효 짝 수</th></tr></thead><tbody id="lagRows"></tbody></table></div>
<div id="lag-H_L" class="plot"></div><div id="lag-L_V" class="plot"></div><div id="lag-H_V" class="plot"></div>
<p class="note">E는 엔트로피 감소량, D는 1인당 손실, A는 변동 크기의 증가율이야. 사건 보기에서는 확인된 사건의 시작을 1, 판정 가능한 다른 라운드를 0으로 둬. 기준 이력이나 확인 구간이 부족한 값은 제외해. 두 방식 모두 같은 회차와 분석 구간 안에서 짝지어. ‘최대’는 조사한 시차 중 |r|가 가장 큰 곳이야. 구간이 겹치는 이동 통계량이라 인접 시차의 값은 서로 관련될 수 있어.</p></section>
<div id="methodHost">{methods(c)}</div>
{'<section><h2>완료한 회차</h2><div class="actions" id="reports"></div></section>' if live else ''}
</main>"""
    script = 'const LIVE = '+('true' if live else 'false')+';\nconst BOOT = '+wire_json(data)+';\n'+COMPLETION_UI+'\n'+CHART+'\n'+TIMELINE+'\n'+UI
    return shell('전략 교체 방식 비교 · 소수파 게임', body, script)

class LiveExperiment:
    def __init__(self, c, folder, engine, use_numpy, quick=False):
        self.c, self.folder, self.engine, self.use_numpy = c, folder, engine, use_numpy
        self.quick = quick
        self.condition = threading.Condition()
        self.run = 1
        self.frame_due = True
        self.next_frame = 0.
        self.new_pair(1)
        self.status, self.error = 'idle', ''
        self.token = secrets.token_urlsafe(24)
        self.speed, self.auto_next = 0, True
        self.speed_revision = 0
        self.next_round_due = None
        self.goal = None
        self.closed = False
        self.completed, self.rows = [], []
        self.completion_html = None
        self.archive = {'busy':False,'message':'','url':''}
        self.worker = threading.Thread(target=self.work, name='paired-minority', daemon=True)
        self.worker.start()

    def new_pair(self, run):
        self.run = run
        self.pair = PairedRun(self.c,run,self.use_numpy,capture=lambda t: self.frame_due,
                              directory=self.folder)
        self.rate_start = time.monotonic()
        self.rate_round = 0
        self.measured_speed = 0.
        self.next_frame = 0.

    def command(self, data):
        with self.condition:
            cmd = data.get('action')
            if cmd == 'settings':
                if self.status != 'idle':
                    raise ValueError('설정은 첫 라운드를 시작하기 전에 바꿀 수 있어.')
                values = asdict(self.c)
                fields = {'n': int, 'q': int, 'rounds': int, 'repeats': int, 'seed': int,
                          'burn': int, 'bins': int, 'mutation_prob': float, 'mutation_sd': float,
                          'probe_round':int,'chain_horizon':int}
                for k, v in data.get('config', {}).items():
                    if k not in fields:
                        raise ValueError('지원하지 않는 설정이야.')
                    values[k] = fields[k](v)
                config = Config(**values)
                validate(config)
                if config.q > 12:
                    raise ValueError('실시간 화면은 선택지 12개까지 지원해.')
                if config != self.c:
                    self.c = config
                    self.new_pair(1)
                    write_settings(self.folder, self.c, self.engine, self.quick)
            elif cmd == 'archive':
                if self.archive['busy']:
                    raise ValueError('기존 결과를 분석하고 있어.')
                path = str(data.get('path') or '').strip().strip('"')
                if not path:
                    raise ValueError('기존 결과 폴더의 경로를 입력해 줘.')
                source = archive_source(path)
                destination = Path(tempfile.mkdtemp(prefix='archive_',dir=self.folder))
                self.archive = {'busy':True,'message':'기존 결과를 여는 중','url':''}
                threading.Thread(target=self.run_archive,args=(source,destination),daemon=True).start()
            elif cmd == 'speed':
                value = data.get('value')
                try:
                    rate = 0.0 if value == 'max' else float(value)
                except (ValueError, TypeError, OverflowError):
                    rate = math.nan
                if isinstance(value, bool) or not math.isfinite(rate) or rate < 0:
                    raise ValueError('0보다 큰 초당 라운드 수를 입력하거나 최고속을 선택해 줘.')
                self.speed = rate
                self.speed_revision += 1
                self.next_round_due = None
            elif cmd == 'auto':
                self.auto_next = bool(data.get('value'))
            elif cmd in ('start', 'step', 'measure', 'finish'):
                if self.status not in ('idle', 'paused', 'between', 'running'):
                    raise ValueError('이 회차의 계산 상태를 확인해 줘.')
                if self.status == 'idle' and data.get('config'):
                    self.command({'action':'settings','config':data['config']})
                if self.status == 'between' or self.run in self.completed:
                    self.new_pair(self.run+1)
                if cmd == 'measure' and self.pair.round >= self.c.burn:
                    raise ValueError('이미 초기 적응기를 지났어.')
                self.goal = {'step': self.pair.round+1, 'measure': self.c.burn,
                             'finish': self.c.rounds}.get(cmd)
                if self.status != 'running' or cmd == 'step':
                    self.next_round_due = None
                    self.next_frame = 0.
                    self.rate_start, self.rate_round = time.monotonic(), self.pair.round
                    self.measured_speed = 0.
                self.status = 'running'
            elif cmd == 'pause':
                if self.status == 'running':
                    self.status, self.goal = 'paused', None
            elif cmd == 'stop':
                self.closed, self.status = True, 'closed'
            else:
                raise ValueError('알 수 없는 명령이야.')
            self.condition.notify_all()

    def work(self):
        try:
            next_yield = time.monotonic()+.02
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.status == 'running')
                    if self.closed:
                        return
                    if self.run in self.completed:
                        self.new_pair(self.run+1)
                    now = time.monotonic()
                    if self.speed:
                        due = self.next_round_due if self.next_round_due is not None else now
                        ahead = due-now
                        # 고속에서는 4밀리초 이내로 묶어서 진행해 라운드마다 OS 타이머를 기다리지 않는다.
                        lead = .004 if self.speed>=500 else 0.
                        if ahead>lead:
                            self.condition.wait(timeout=min(ahead,threading.TIMEOUT_MAX))
                            continue
                        self.next_round_due = max(due,now)+1/self.speed
                    self.frame_due = (now>=self.next_frame or self.goal==self.pair.round+1
                                      or self.pair.round+1==self.c.rounds)
                    if self.frame_due:
                        self.next_frame = now+.15
                    self.pair.advance()  # 두 방식이 같은 라운드를 마친 뒤에만 상태를 공개한다.
                    finished = self.pair.round == self.c.rounds
                    at_goal = self.goal is not None and self.pair.round >= self.goal
                    elapsed = now-self.rate_start
                    if elapsed>=.8 or (finished and elapsed>0):
                        self.measured_speed = (self.pair.round-self.rate_round)/elapsed
                        self.rate_start,self.rate_round = now,self.pair.round
                    if finished:
                        self.status = 'analyzing'
                    elif at_goal:
                        self.status, self.goal = 'paused', None
                        self.condition.notify_all()
                if now>=next_yield:
                    time.sleep(0)  # 상태 조회와 속도 명령이 두 라운드 사이에 진입할 기회를 준다.
                    next_yield = time.monotonic()+.02
                if finished:
                    rows = report_pair(self.folder, self.c, self.run, self.pair, self.engine, self.use_numpy)
                    overview = write_index(self.folder, self.c, self.rows+rows, self.engine, live=True)
                    with self.condition:
                        self.rows.extend(rows)
                        self.completed.append(self.run)
                        self.completion_html = overview
                        if self.closed:
                            return
                        self.goal = None
                        if self.run >= self.c.repeats:
                            self.status = 'done'
                        else:
                            self.status = 'running' if self.auto_next and not at_goal else 'between'
                        self.condition.notify_all()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self.condition:
                self.status, self.error = 'error', str(exc)
                self.condition.notify_all()

    def run_archive(self,source,destination):
        def progress(message):
            with self.condition:
                self.archive['message'] = message
        try:
            page = analyze_archive(source,destination,progress)
            result = {'busy':False,'message':'기존 결과 분석 완료','url':'/files/'+destination.name+'/'+page.name}
        except Exception as exc:
            result = {'busy':False,'message':'기존 결과 분석 오류 · '+str(exc),'url':''}
        with self.condition:
            self.archive = result

    def state(self, client_run=0, since=0, seen=0):
        with self.condition:
            n = self.pair.round
            start = since if client_run==self.run and 0<=since<=n else 0
            modes = {}
            for m in MODES:
                tr, meter = self.pair.traces[m], self.pair.meters[m]
                if self.pair.tables:
                    increments = {}
                    for k,group in meter.events.items():
                        # 확인 라운드 기준으로 전달해야 이전 요청 이후 확인된 L/V도 빠지지 않는다.
                        a,b = group.between(start-group.delay,n)
                        increments[k] = list(group.starts[a:b])
                    modes[m] = {'series':{},'events':increments,'view':self.pair.views[m]}
                else:
                    modes[m] = {'series': {k: tr[k][start:n] if n else [] for k in (*SERIES,'counts')},
                                'events': {k: v[:] for k, v in meter.events.items()}, 'view': self.pair.views[m]}
            return {'config': asdict(self.c), 'engine': self.engine, 'run': self.run, 'round': n,
                    'start': start, 'modes': modes, 'status': self.status, 'error': self.error,
                    'definitions': methods(self.c) if start==0 else None,
                    'auto_next': self.auto_next, 'speed': self.speed, 'completed': self.completed[:], 'archive':dict(self.archive),
                    'large':bool(self.pair.tables), 'measured_speed':self.measured_speed, 'speed_revision':self.speed_revision,
                    'completion_html': self.completion_html if seen != len(self.completed) else None}

    def correlations(self, run, kind, phase, basis='continuous'):
        if kind not in ENTROPIES or phase not in PHASES or basis not in ('continuous','events'):
            raise ValueError('분석 구간과 엔트로피 종류를 확인해 줘.')
        with self.condition:
            if run != self.run:
                raise ValueError('회차가 바뀌었어. 현재 회차에서 다시 계산해 줘.')
            c, n = self.c, self.pair.round
            pair = self.pair
            if getattr(pair,'final_phases',None) is not None:
                series = 'event_curves' if basis=='events' else 'curves'
                return {'run':run,'round':n,'kind':kind,'phase':phase,'basis':basis,
                        'modes':{m:pair.final_phases[m][phase][kind][series] for m in MODES}}
            if pair.tables:
                for table in pair.tables.values():
                    table.flush()
                traces = pair.traces
                events = {m:pair.meters[m].events if basis=='events' else None for m in MODES}
            else:
                traces = {m: {k: pair.traces[m].get(k, [])[:] for k in
                              ('drop_'+kind, 'loss_degree', 'amp_pct')} for m in MODES}
                events = {m:{k:v[:] for k,v in pair.meters[m].events.items()}
                          for m in MODES} if basis=='events' else dict.fromkeys(MODES)
        if pair.tables:
            curves = {m:disk_phase_curves(traces[m],c,phase,kind,self.use_numpy,None,events[m],n)
                      for m in MODES}
        else:
            curves = {m:phase_curves(traces[m],c,phase,kind,self.use_numpy,events=events[m]) for m in MODES}
        return {'run':run,'round':n,'kind':kind,'phase':phase,'basis':basis,'modes':curves}

    def disk_snapshot(self, run, lo, hi):
        with self.condition:
            pair = self.pair
            if run!=self.run or not pair.tables or not 1<=lo<=hi<=pair.round:
                raise ValueError('회차 또는 라운드 범위를 다시 확인해 줘.')
            for table in pair.tables.values():
                table.flush()
            return pair

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join(timeout=3)


def write_settings(folder, c, engine, quick):
    dump_json(folder/'config.json', {'version': VERSION, 'config': asdict(c), 'engine': engine,
        'python': platform.python_version(), 'quick': quick, 'modes': list(MODES), 'paired': True,
        'replacement_rules': {'random': 'uniform_other_copy', 'best': 'best_other_copy_with_optional_mutation'}})
    (folder/'README.txt').write_text(readme(c,engine,quick),encoding='utf-8')


def live_html(app):
    return dashboard(app.c, {'config': asdict(app.c), 'engine': app.engine, 'token': app.token},
                     live=True)

def make_live_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_data(self,data,status=200,mime='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            if isinstance(data,str):data=data.encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def send_samples(self, q):
            run,lo,hi,step = (int(q[k][0]) for k in ('run','lo','hi','stride'))
            mode = q['mode'][0]
            if mode not in MODES or step < 1 or (hi-lo)//step+1 > 16384:
                raise ValueError('그래프 표시 범위를 확인해 줘.')
            pair = app.disk_snapshot(run,lo,hi)
            return self.send_data(pair.tables[mode].samples(lo,hi,step),mime='application/octet-stream')

        def export_trace(self, q):
            run,lo,hi = (int(q[k][0]) for k in ('run','lo','hi'))
            kind = q['kind'][0]
            if kind not in ENTROPIES:
                raise ValueError('엔트로피 종류를 확인해 줘.')
            pair = app.disk_snapshot(run,lo,hi)
            keys = ('h_'+kind,'drop_'+kind,'loss_degree','amp_pct','volatility','slope','total')
            cols = [TRACE_KEYS.index(k) for k in keys]
            self.send_response(200)
            self.send_header('Content-Type','text/csv; charset=utf-8')
            self.send_header('Content-Disposition',f'attachment; filename="paired-trace-{run}.csv"')
            self.end_headers()
            self.wfile.write(b'\xef\xbb\xbf')
            batch = io.StringIO(newline='')
            writer = csv.writer(batch)
            writer.writerow(['round',*[m+'_'+k for m in MODES for k in keys]])
            random,best = (pair.tables[m] for m in MODES)
            for (start,rdata),(_,bdata) in zip(random.blocks(lo-1,hi,2048),best.blocks(lo-1,hi,2048)):
                pairs = zip(random.record.iter_unpack(rdata),best.record.iter_unpack(bdata))
                for i,rows in enumerate(pairs,start+1):
                    writer.writerow([i,*[None if math.isnan(row[j]) else row[j] for row in rows for j in cols]])
                self.wfile.write(batch.getvalue().encode('utf-8'))
                batch.seek(0)
                batch.truncate()

        def do_GET(self):
            url=urlsplit(self.path)
            if url.path=='/':return self.send_data(live_html(app),mime='text/html; charset=utf-8')
            if url.path in ('/api/state','/api/correlation','/api/samples','/api/trace.csv'):
                q = parse_qs(url.query)
                try:
                    run = int(q.get('run', ['0'])[0])
                    if url.path == '/api/samples':
                        return self.send_samples(q)
                    if url.path == '/api/trace.csv':
                        return self.export_trace(q)
                    if url.path == '/api/correlation':
                        return self.send_data(app.correlations(run, q.get('kind',['strategy'])[0], q.get('phase',['all'])[0], q.get('basis',['continuous'])[0]))
                    return self.send_data(app.state(run, int(q.get('since',['0'])[0]), int(q.get('seen',['0'])[0])))
                except (BrokenPipeError,ConnectionResetError):
                    return
                except (ValueError, TypeError, KeyError) as exc:
                    return self.send_data({'error':str(exc)}, 400)
            if url.path.startswith('/files/'):
                base=app.folder.resolve();relative=unquote(url.path[len('/files/'):])
                target=(base/relative).resolve()
                try:target.relative_to(base)
                except ValueError:return self.send_data({'error':'허용하지 않는 경로'},403)
                if not target.is_file():return self.send_data({'error':'아직 생성되지 않은 결과'},404)
                with target.open('rb') as contents:
                    self.send_response(200)
                    self.send_header('Content-Type',mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Length',str(target.stat().st_size))
                    self.end_headers()
                    try:
                        shutil.copyfileobj(contents,self.wfile,256*1024)
                    except (BrokenPipeError,ConnectionResetError):
                        pass
                return
            if url.path=='/favicon.ico':return self.send_data(b'',204)
            self.send_data({'error':'없는 주소'},404)
        def do_POST(self):
            if self.path!='/api/control':return self.send_data({'error':'없는 주소'},404)
            if not secrets.compare_digest(self.headers.get('X-Experiment-Token',''),app.token):
                return self.send_data({'error':'실험 화면에서 조작해 줘.'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8192:raise ValueError('요청 크기를 확인해 줘.')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('요청 형식을 확인해 줘.')
                app.command(data)
                with app.condition:
                    reply = {'ok':True,'speed':app.speed,'speed_revision':app.speed_revision,'rounds':app.c.rounds}
                self.send_data(reply)
                if data.get('action')=='stop':
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
            except (ValueError,TypeError,KeyError) as exc:self.send_data({'error':str(exc)},400)
    return Handler


def serve_live(c,folder,engine,use_numpy,no_open=False,quick=False):
    app=LiveExperiment(c,folder,engine,use_numpy,quick)
    try:
        server=ThreadingHTTPServer(('127.0.0.1',0),make_live_handler(app))
    except OSError:
        app.close()
        raise SystemExit('로컬 웹 서버를 열 수 없어. 다른 프로그램/방화벽 설정을 확인하거나 --batch로 실행해 줘.')
    server.daemon_threads=True
    url=f'http://127.0.0.1:{server.server_address[1]}/'
    print('\n실시간 실험 웹사이트:',url,flush=True)
    print('브라우저에서 두 방식 시작을 눌러 줘. Python 창은 열어 둬. 종료는 웹 화면의 실행 종료 버튼.',flush=True)
    if not no_open:
        try:webbrowser.open(url)
        except Exception:print('위 주소를 브라우저에 직접 붙여 넣어 줘.',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:print('\n실험 종료',flush=True)
    finally:app.close();server.server_close()

def check_evidence(use_numpy):
    """관측 분모·시간 경계, 원래 경로 복제, 개입 내용, 저장 기록 재생을 비교한다."""
    import itertools
    check = Config(n=33,rounds=90,repeats=3,burn=12,h_window=5,loss_base=6,
                   loss_block=2,loss_repeats=2,vol_window=4,vol_hold=2,chain_horizon=30)
    for plus,minus,expected in ((0,0,1.),(3,3,1.),(5,0,.0625),(0,6,.03125)):
        assert math.isclose(paired_binary_p(plus,minus),expected,abs_tol=1e-12)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for m in MODES:
            (folder/m).mkdir()
        sample = [
            {'H_strategy':[10,20,38],'L':[11,20,24,31],'V':[14,21,25,31]},
            {'H_strategy':[10,30],'L':[11,22,33],'V':[10,12,25,35]},
            {'H_strategy':[30],'L':[8,20,25,31],'V':[10,13,22,36]}]
        c = replace(check,rounds=40,chain_horizon=5,loss_block=1,loss_repeats=2)
        for i,e in enumerate(sample,1):
            with (folder/'best'/f'run_{i:02d}_events.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.writer(f)
                w.writerow(['event','start_round','confirmed_round'])
                for kind,vals in e.items():
                    w.writerows((kind,t,t+(0 if kind=='H_strategy' else 1)) for t in vals)
        result = matched_event_results(c,folder,'best',[1,2,3])
        for edge,source in (('HL','H_strategy'),('HLV','H_strategy'),('LV','L')):
            on = []; controls = []; short = unmatched = 0
            def has(e,t):
                after = t+int(edge=='LV')
                endpoint = after+5
                if edge=='HL':
                    return any(after<l<=endpoint for l in e['L'])
                if edge=='LV':
                    return any(after<v<=endpoint for v in e['V'])
                return any(after<l<v<=endpoint for l,v in itertools.product(e['L'],e['V']))
            for e in sample:
                for t in e[source]:
                    if t+int(edge=='LV')+5+1>40:
                        short+=1
                        continue
                    others=[x for x in sample if t not in x[source]]
                    if not others:
                        unmatched+=1
                        continue
                    on.append(has(e,t))
                    controls.append(sum(has(x,t) for x in others)/len(others))
            r=result[edge]
            assert (r['n'],r['hits'],r['short'],r['unmatched'])==(len(on),sum(on),short,unmatched)
            assert math.isclose(r['control_sum'],sum(controls),abs_tol=1e-12)
        single = matched_event_results(c,folder,'best',[1])['HLV']
        assert single['n']==0 and single['unmatched'] and single['short']
        for engine in ((False,True) if use_numpy else (False,)):
            pair = PairedRun(check,1,engine,capture=True)
            at,_,_ = probe_schedule(check)
            for _ in range(check.rounds):
                pair.advance()
                if pair.round==at:
                    views = deepcopy(pair.views)
            for m in MODES:
                study = pair.studies[m]
                reference = recorded_probe(check,pair.traces[m],pair.meters[m].events)
                result = study.calculate(engine,reference)
                assert result['arms']['control']==reference
                assert result==study.calculate(engine,reference)
                gen = rounds_numpy if engine else rounds_python
                members = list(range(check.n//2))
                sim,history,_ = study.saved
                def first(arm):
                    plan = (arm,at,at+3,members,1)
                    it = gen(check,m,1,False,True,deepcopy(history),initial=sim,intervention=plan)
                    return next(it)[1]
                base,uniform,crowd = first('control'),first('homogeneous'),first('crowd')
                assert base==views[m] and all(crowd['actions'][i]==1 for i in members)
                tape=[]
                crowded=gen(check,m,1,False,True,deepcopy(history),initial=sim,intervention=('crowd',at,at+3,members,1,tape))
                protected=gen(check,m,1,False,True,deepcopy(history),initial=sim,intervention=('loss_shield',at,at+3,members,1,tape))
                for tick in range(4):
                    real=next(crowded)[1]; relief=next(protected)[1]
                    assert real['actions']==relief['actions'] and real['winner']==relief['winner']
                    if tick==0:
                        for idx,choice in enumerate(real['actions']):
                            expected=0. if choice==real['winner'] else 1/(check.q-1)
                            assert math.isclose(relief['scores'][idx]-real['scores'][idx],expected,abs_tol=1e-12)
                original = base['probabilities']; changed = uniform['probabilities']
                assert changed[len(members):]==original[len(members):]
                for j in range(check.q):
                    assert math.isclose(sum(v[j] for v in original),sum(v[j] for v in changed),abs_tol=1e-10)
                    assert all(v[j]==changed[0][j] for v in changed[:len(members)])
                stem=folder/m/'run_01_trace.csv'
                write_trace(stem,pair.traces[m],check)
                rebuilt=replay_archived_pair(check,m,1,engine,folder)
                assert rebuilt['arms']==result['arms']
                with stem.open(encoding='utf-8-sig',newline='') as f:
                    rows=list(csv.reader(f))
                rows[1][rows[0].index('count_1')]='-1'
                with stem.open('w',encoding='utf-8-sig',newline='') as f:
                    csv.writer(f).writerows(rows)
                assert 'arms' not in replay_archived_pair(check,m,1,engine,folder)
            long = replace(check,rounds=LARGE_ROUNDS)
            disk = PairedRun(long,1,engine,directory=folder)
            try:
                for _ in range(probe_schedule(long)[1]):
                    disk.advance()
                for m in MODES:
                    disk.tables[m].flush()
                    expected=recorded_probe(long,disk.traces[m],disk.meters[m].events)
                    assert disk.studies[m].calculate(engine,expected)['arms']['control']==expected
            finally:
                for f in disk.tables.values():
                    f.close()
    print('점검 통과 · 관측 대조 · 대조 경로 재현 · 개입 처치 · 과거 기록 검증 · 대용량 기록')


def check_large_settings(use_numpy):
    """웹 요청과 동일한 명령으로 1천만 회 목표를 시작하고 첫 513회를 대조한다."""
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for mode in MODES:
            (folder/mode).mkdir()
        c = Config(n=31,rounds=30000,repeats=1,burn=512,max_lag=10,lag_step=5)
        app = LiveExperiment(c,folder,'자가 점검',use_numpy)
        try:
            app.command({'action':'settings','config':{'rounds':'10000000'}})
            app.command({'action':'speed','value':'100000'})
            app.command({'action':'measure'})
            with app.condition:
                ready=app.condition.wait_for(lambda:app.status in ('paused','error'),timeout=30)
            if not ready or app.status!='paused':
                raise RuntimeError(app.error or '초기 적응 구간 점검 시간 초과')
            app.command({'action':'step'})
            with app.condition:
                app.condition.wait_for(lambda:app.status in ('paused','error'),timeout=5)
            state=app.state()
            if (state['speed'],state['config']['rounds'],state['round'],state['large'])!=(100000,10000000,513,True):
                raise AssertionError('속도·라운드 적용 오류')
            reference = PairedRun(replace(c,rounds=513),1,use_numpy)
            for _ in range(513):
                reference.advance()
            for m in MODES:
                table=app.pair.tables[m]
                table.flush()
                tr=reference.traces[m]
                for offset,block in table.blocks(0,513):
                    for i,row in enumerate(table.record.iter_unpack(block),offset):
                        expected = [tr[k][i] for k in TRACE_KEYS]+tr['counts'][i]
                        if any((not math.isnan(a) if b is None else a!=b) for a,b in zip(row,expected)):
                            raise AssertionError(f'{m} {i+1}라운드 원자료 불일치')
                for k,group in app.pair.meters[m].events.items():
                    assert list(group)==reference.meters[m].events[k], (m,k)
                for basis in ('continuous','events'):
                    all_events=reference.meters[m].events if basis=='events' else None
                    standard=phase_curves(tr,c,'all','strategy',use_numpy,events=all_events)
                    compact=app.correlations(1,'strategy','all',basis)['modes'][m]
                    for key in standard:
                        for a,b in zip(standard[key],compact[key]):
                            if a['n']!=b['n'] or a['lag']!=b['lag'] or ((a['r'] is None)!=(b['r'] is None)):
                                raise AssertionError(f'{m} {key} 유효 짝 불일치')
                            if a['r'] is not None and not math.isclose(a['r'],b['r'],abs_tol=1e-10):
                                raise AssertionError(f'{m} {key} 상관계수 불일치')
            print('점검 통과 · 설정 100,000라운드/초 · 목표 10,000,000라운드 · 두 방식의 첫 513라운드와 사건·상관 일치')
        finally:
            app.close()
            for table in app.pair.tables.values():
                table.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quick',action='store_true',help='작은 조건으로 빠른 실행')
    p.add_argument('--batch',action='store_true',help='웹 조작 없이 전체를 자동 계산')
    p.add_argument('--evidence-from',type=Path,help='기존 결과 폴더에서 관측 대조 및 검증된 개입 분석')
    p.add_argument('--check-evidence',action='store_true',help='관측 대조·개입 분기 재현성 점검')
    p.add_argument('--check-large',action='store_true',help='대용량 속도·라운드 설정과 원자료 대조')
    p.add_argument('--engine',choices=['auto','python','numpy'],default='auto')
    p.add_argument('--no-open',action='store_true')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'minority_results')
    for name in ('rounds','repeats','burn','agents','seed','bins','max-lag','lag-step','probe-round','chain-horizon'):
        p.add_argument('--'+name,type=int)
    p.add_argument('--mutation-prob',type=float)
    p.add_argument('--mutation-sd',type=float)
    args=p.parse_args()
    c=Config(**asdict(DEFAULT))
    if args.quick:
        c.n,c.rounds,c.repeats,c.burn=101,3000,3,1000
        c.chain_horizon=1000
    for arg,field in [('rounds','rounds'),('repeats','repeats'),('burn','burn'),('agents','n'),
                      ('seed','seed'),('bins','bins'),('max_lag','max_lag'),('lag_step','lag_step'),
                      ('mutation_prob','mutation_prob'),('mutation_sd','mutation_sd'),
                      ('probe_round','probe_round'),('chain_horizon','chain_horizon')]:
        if getattr(args,arg) is not None:
            setattr(c,field,getattr(args,arg))
    validate(c)
    use_numpy=False
    if args.engine!='python':
        try:
            import numpy as np
            np.random.default_rng(0).random()
            use_numpy=True
        except (ImportError,OSError):
            if args.engine=='numpy':
                raise SystemExit('NumPy를 사용할 수 없어. --engine python으로 실행해 줘.')
    if args.check_large:
        return check_large_settings(use_numpy)
    if args.check_evidence:
        return check_evidence(use_numpy)
    if args.evidence_from:
        source = archive_source(args.evidence_from)
        args.out.mkdir(parents=True,exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix='chain_evidence_',dir=args.out)).resolve()
        report = analyze_archive(source,destination,lambda m:print(m,flush=True))
        print('분석 결과:',report,flush=True)
        if not args.no_open:
            webbrowser.open(report.as_uri())
        return destination
    engine=f'NumPy {np.__version__} / PCG64' if use_numpy else f'Python {platform.python_version()} / MT19937'
    try:
        args.out.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_'),dir=args.out)).resolve()
    except OSError:
        folder=Path(tempfile.mkdtemp(prefix='minority_results_')).resolve()
    for mode in MODES:(folder/mode).mkdir()
    write_settings(folder,c,engine,args.quick)
    print('결과 폴더:',folder,flush=True)
    print(f'{engine} · {c.n}명 · {c.rounds:,}라운드 · 방식별 {c.repeats}회',flush=True)
    if not args.batch:
        serve_live(c, folder, engine, use_numpy, args.no_open, args.quick)
        return folder
    rows = []
    started = time.monotonic()
    for run in range(1,c.repeats+1):
        print(f'[{run}/{c.repeats}] 랜덤과 베스트 동시 실행', flush=True)
        pair = PairedRun(c, run, use_numpy, directory=folder)
        for _ in range(c.rounds):
            pair.advance()
            if pair.round % 5000 == 0:
                print(f'  두 방식 {pair.round:,}/{c.rounds:,}라운드', flush=True)
        rows.extend(report_pair(folder,c,run,pair,engine,use_numpy))
        write_index(folder,c,rows,engine)
    report=folder/'index.html'
    print(f'완료 ({time.monotonic()-started:.1f}초): {report}',flush=True)
    if not args.no_open:
        try:webbrowser.open(report.as_uri())
        except Exception:print('위 index.html을 직접 열어 줘.',flush=True)
    return folder


if __name__=='__main__':
    main()
