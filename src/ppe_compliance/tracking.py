"""Session-local geometric tracking and conservative status-change events.

This is not appearance re-identification. Ambiguous ownership starts new review
tracks rather than borrowing another worker's green state.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from ppe_compliance.association import iou


REVIEW = "review required"
PPE = "PPE detected"


@dataclass(frozen=True)
class TrackingRules:
    min_iou: float = .25
    ambiguity_margin: float = .1
    track_ttl_seconds: float = 1.2
    max_observation_gap_seconds: float = .5
    confirmation_observations: int = 3
    confirmation_seconds: float = .2

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"Invalid tracking rule {key}")
        if not 0 < self.min_iou <= 1 or not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("Tracking overlap/margin out of range")
        if type(self.confirmation_observations) is not int or self.confirmation_observations < 1:
            raise ValueError("confirmation_observations must be a positive integer")
        if self.confirmation_seconds < 0 or self.max_observation_gap_seconds <= 0 or self.track_ttl_seconds < self.max_observation_gap_seconds:
            raise ValueError("Invalid tracking time limits")


@dataclass
class Track:
    identifier: str
    box: list
    last_seen: float
    confirmed: str = REVIEW
    candidate: str | None = None
    candidate_since: float = 0.
    observations: int = 0


class WorkerTracker:
    def __init__(self, rules=None):
        self.rules = rules or TrackingRules()
        self.tracks = {}
        self.next_id = 1
        self.last_update = None
        self.event_count = 0
        self.status_change_count = 0
        self.history = deque(maxlen=12)

    def _event(self, events, kind, track, at, previous=None, reason=None):
        self.event_count += 1
        if kind == "status_changed":
            self.status_change_count += 1
        event = {"event_id":self.event_count,"type":kind,"track_id":track.identifier,
                 "time_seconds":round(at,6),"previous_status":previous,
                 "status":track.confirmed,"reason":reason}
        events.append(event)
        self.history.append(event)

    def _reset(self, track, events, at, reason):
        previous = track.confirmed
        track.confirmed,track.candidate,track.observations = REVIEW,None,0
        if previous != REVIEW:
            self._event(events,"status_changed",track,at,previous,reason)

    def update(self, assessment, at):
        if isinstance(at,bool) or not math.isfinite(at) or (self.last_update is not None and at < self.last_update):
            raise ValueError("Tracking timestamps must be finite and nondecreasing")
        self.last_update = at
        result = deepcopy(assessment)
        workers = result["workers"]
        events = []
        for identifier,track in list(self.tracks.items()):
            gap = at-track.last_seen
            if gap > self.rules.max_observation_gap_seconds:
                self._reset(track,events,at,"PPE evidence expired")
            if gap > self.rules.track_ttl_seconds:
                self._event(events,"track_lost",track,at,reason="Worker no longer observed; ID retired")
                del self.tracks[identifier]
        candidates = {}
        reverse = {}
        for index,worker in enumerate(workers):
            scores = sorted(((iou(worker["bbox_xyxy"],track.box),identifier)
                             for identifier,track in self.tracks.items()),reverse=True)
            scores = [pair for pair in scores if pair[0] >= self.rules.min_iou]
            candidates[index] = scores
            for score,identifier in scores:
                reverse.setdefault(identifier,[]).append((score,index))
        for scores in reverse.values():
            scores.sort(reverse=True)
        assigned = {}
        ambiguous = set()
        for index,scores in candidates.items():
            if not scores:
                continue
            score,identifier = scores[0]
            competitors = reverse[identifier]
            close_owner = len(scores)>1 and score-scores[1][0] <= self.rules.ambiguity_margin
            close_worker = len(competitors)>1 and competitors[0][0]-competitors[1][0] <= self.rules.ambiguity_margin
            if close_owner or close_worker:
                ambiguous.add(index)
                for _,candidate in scores:
                    self._reset(self.tracks[candidate],events,at,"Track identity is ambiguous")
            elif competitors[0][1] == index:
                assigned[index] = identifier
            else:
                # A poorer competing match must not inherit the winning track.
                ambiguous.add(index)
        for index,worker in enumerate(workers):
            identifier = assigned.get(index)
            if identifier is None:
                identifier = f"T{self.next_id:03d}"
                self.next_id += 1
                track = Track(identifier,list(worker["bbox_xyxy"]),at)
                self.tracks[identifier] = track
                self._event(events,"track_started",track,at,reason="New session-local geometric track")
            track = self.tracks[identifier]
            instant = worker["status"]
            if index in ambiguous:
                instant = REVIEW
                worker["reasons"].append("Track association ambiguous; previous PPE state not inherited")
            if track.candidate != instant:
                track.candidate,track.candidate_since,track.observations = instant,at,1
            else:
                track.observations += 1
            confirmed = (track.observations >= self.rules.confirmation_observations and
                         at-track.candidate_since >= self.rules.confirmation_seconds)
            if confirmed and track.confirmed != instant:
                previous = track.confirmed
                track.confirmed = instant
                self._event(events,"status_changed",track,at,previous,"Consistent current observations")
            # Unknown evidence is never displayed as green, even during debounce.
            display = instant if instant != PPE else (PPE if confirmed else REVIEW)
            if display == REVIEW and instant == PPE:
                worker["reasons"].insert(0,"Confirming repeated PPE evidence before showing green")
            worker["frame_worker_id"] = worker["worker_id"]
            worker["worker_id"] = worker["track_id"] = identifier
            worker["instant_status"] = worker["status"]
            worker["status"] = display
            worker["display_label"] = ("Checking PPE"
                                       if instant == PPE and display != PPE else display)
            worker["tracking"] = {"identity_ambiguous":index in ambiguous,"confirmed_status":track.confirmed,
                                   "candidate_status":instant,"candidate_observations":track.observations,
                                   "candidate_seconds":round(at-track.candidate_since,6)}
            track.box,track.last_seen = list(worker["bbox_xyxy"]),at
        result["summary"] = {"workers":len(workers), **{status:sum(w["status"]==status for w in workers)
                              for status in (PPE,"potential violation",REVIEW)}}
        result["status_events"] = events
        result["tracking_rules"] = asdict(self.rules)
        result["worker_id_scope"] = "Session-local geometric tracks; not verified identity. Crossings/occlusion can change IDs."
        return result

    def summary(self):
        return {"tracks_created":self.next_id-1,"active_tracks":len(self.tracks),
                "events":self.event_count,"confirmed_status_changes":self.status_change_count,
                "rules":asdict(self.rules)}
