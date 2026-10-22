"""Python 3.7 compatible, transport independent replay protection for the tick owner."""
import json


class ProtocolError(RuntimeError):
    pass


class EpisodeProtocol:
    def __init__(self):
        self.episode = None
        self.seen_episodes = set()
        self.last_request = -1
        self.cache = {}

    def execute(self, request, handler):
        rid, episode, op = request.get("request_id"), request.get("episode_id"), request.get("op")
        if not isinstance(rid, int) or rid < 0 or not isinstance(episode, str):
            raise ProtocolError("request_id integer and episode_id string required")
        fingerprint = json.dumps(request, sort_keys=True)
        if episode == self.episode and rid in self.cache:
            old, response = self.cache[rid]
            if old != fingerprint:
                raise ProtocolError("same request_id with different payload")
            return response
        if rid <= self.last_request:
            raise ProtocolError("stale request_id")
        if op == "begin":
            if episode in self.seen_episodes:
                raise ProtocolError("stale episode_id")
        elif episode != self.episode:
            raise ProtocolError("stale episode_id")
        # Mark before execution. A failed tick may have reached the server; never retry it.
        self.last_request = rid
        if op == "begin":
            self.episode = episode
            self.seen_episodes.add(episode)
            self.cache = {}
        try:
            response = {"ok": True, **handler(request)}
        except Exception as exc:
            response = {"ok": False, "error": type(exc).__name__ + ": " + str(exc),
                        "infrastructure_failure": True}
        response.update({"episode_id": episode, "request_id": rid})
        self.cache[rid] = (fingerprint, response)
        # Current episode has at most 900 controls; bounded protection still permits retries.
        if len(self.cache) > 4096:
            self.cache.pop(min(self.cache))
        return response
