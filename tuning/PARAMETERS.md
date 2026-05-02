# Tracker parameters reference

Full description of every key in [`tracker_config.json`](../tracker_config.json).
For tuning workflow see [HOWTO.md](HOWTO.md); this file is the reference.

The tracker pipeline has three stages, and each parameter belongs to one of them:

```
   ┌────────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────┐
   │  YOLO      │ →  │ deduplicate    │ →  │ IOU tracker  │ →  │ Counter │
   │  detector  │    │ overlapping    │    │ assigns IDs  │    │ checks  │
   │            │    │ boxes          │    │ across frames│    │ crossings│
   └────────────┘    └────────────────┘    └──────────────┘    └─────────┘
        ↑                  ↑                     ↑                  ↑
     confidence         iou,                  min_iou,           min_hits,
                        deduplicate           max_distance,      buffer
                                              max_disappeared
```

---

## `confidence` — detector confidence threshold

**Default:** `0.3`
**Type:** float in `[0.0, 1.0]`
**Stage:** detector
**Code:** [`run_yolo11_tracking.py:789`](../run_yolo11_tracking.py:789) — passed as `conf_threshold` into `session.detect(...)`.

Boxes with predicted class probability below this value are discarded by the detector before they ever reach the tracker.

| Push UP (e.g. 0.3 → 0.5) | Push DOWN (e.g. 0.3 → 0.2) |
|---|---|
| Fewer false positives | Catches faint / partially occluded objects |
| Less flicker → more stable tracks | More flicker, more ghost tracks |
| Risk: misses real objects in poor light | Risk: sees vehicles in shadows that aren't there |

**Sensible range:** 0.2–0.5. Below 0.2 you get noise; above 0.6 you start losing real distant vehicles.

**Interacts with:** `min_hits` — both filter weak detections, but at different stages. Bumping `confidence` is more surgical (kills bad detections at the source); bumping `min_hits` is more lenient (lets weak detections in but doesn't count them until they prove themselves).

---

## `iou` — NMS / dedup IoU threshold

**Default:** `0.45`
**Type:** float in `[0.0, 1.0]`
**Stage:** detector + dedup
**Code:** Two places:
1. [`run_yolo11_tracking.py:790`](../run_yolo11_tracking.py:790) — passed as `iou_threshold` to detector for **non-maximum suppression** (NMS): when two boxes overlap by more than this, the lower-confidence one is dropped.
2. [`run_yolo11_tracking.py:803`](../run_yolo11_tracking.py:803) — passed to `deduplicate_detections()` for cross-class dedup (e.g. "car" + "truck" both predicted on the same vehicle).

| Push UP (0.45 → 0.6) | Push DOWN (0.45 → 0.3) |
|---|---|
| Less aggressive suppression — keeps more overlapping boxes | More aggressive — kills more overlaps |
| Risk: duplicate boxes on one vehicle survive | Risk: real adjacent vehicles get merged |

**Sensible range:** 0.4–0.5. Rarely worth tuning — the YOLO default is well-calibrated. Touch only if you see obvious double-boxing or merging.

---

## `min_iou` — tracker association IoU floor

**Default:** `0.15`
**Type:** float in `[0.0, 1.0]`
**Stage:** tracker
**Code:** [`run_yolo11_tracking.py:190`](../run_yolo11_tracking.py:190) — `if max_iou < self.min_iou: ...` — when matching detections to existing tracks, the IoU must be at least this much.

How tracking works: for every existing track, compute IoU against every new detection. The Hungarian-like assignment uses the highest IoU match — but only if it clears `min_iou`. Below that, the detection is treated as a *new* track.

| Push UP (0.15 → 0.30) | Push DOWN (0.15 → 0.10) |
|---|---|
| Stricter matching — only well-overlapping detections inherit the track ID | Looser matching — distant detections can re-attach to tracks |
| Track IDs change less often mid-frame | Lost tracks recover faster |
| Risk: real fast vehicles get a new ID every few frames (overcounting) | Risk: a new vehicle entering near an old track gets the old ID |

**Sensible range:** 0.10–0.30. Default 0.15 is intentionally low because vehicle bounding boxes jitter — too high and a single fast car gets 3 different IDs.

**Interacts with:** `max_distance` — these are the two matching criteria. IoU first, then centroid distance as fallback. If both are too strict, no matches; if both too loose, ID swaps.

---

## `max_distance` — centroid distance fallback

**Default:** `200.0`
**Type:** float, **pixels**
**Stage:** tracker
**Code:** [`run_yolo11_tracking.py:242`](../run_yolo11_tracking.py:242) — `if min_dist > self.max_distance: ...` — when IoU matching fails (no detection overlaps an existing track enough), the tracker falls back to matching by centroid distance. If the closest detection is further than this, the track stays unmatched.

This is what saves fast-moving vehicles whose bounding boxes don't overlap between consecutive frames.

| Push UP (200 → 300) | Push DOWN (200 → 100) |
|---|---|
| Rescues fast vehicles that "jump" between frames | Stricter — won't connect distant detections to old tracks |
| Risk: a *new* vehicle near a lost track gets the old ID | Risk: fast vehicles get fragmented into multiple tracks |

**Sensible range:** 100–300 pixels (depends on frame size and scene speed).
- Slow indoor / parking-lot footage: 100–150
- Highway from a fixed camera: 200–300
- Drone footage / fast pans: even higher

**Interacts with:** frame resolution. The default `200` was tuned at 1024×768. At 640×480 you'd want ~120; at 1920×1080 you'd want ~350. **This isn't normalized**, so check what resolution your scripts actually capture (`--input-small` etc.).

---

## `max_disappeared` — frames before deleting a lost track

**Default:** `50`
**Type:** int, **frames**
**Stage:** tracker
**Code:** [`run_yolo11_tracking.py:159`](../run_yolo11_tracking.py:159) and [`:278`](../run_yolo11_tracking.py:278) — when a track isn't matched to any detection in a frame, its `disappeared` counter increments. Once it exceeds `max_disappeared`, the track is deleted.

Translates to **time** via frame rate: at 30 fps, `max_disappeared = 50` ≈ 1.7 seconds of grace period.

| Push UP (50 → 100) | Push DOWN (50 → 20) |
|---|---|
| Tracks survive longer occlusions (passing behind a tree, briefly out of frame) | Lost tracks die faster — frees up IDs |
| Risk: a stale track getting matched to a *different* vehicle 3 sec later | Risk: brief occlusions break tracks → fragmented counts |

**Sensible range:** 20–100 frames. Match it to your scene:
- Clean line of sight, no occluders: 20–30 is fine.
- Trees, poles, parked cars between camera and traffic: 60–100.

**Interacts with:** `max_distance` — when a track survives a long disappearance, on reappearance it'll match by distance. Big `max_disappeared` + small `max_distance` is safe; big `max_disappeared` + big `max_distance` risks ID swaps.

---

## `min_hits` — frames before counting crossings

**Default:** `3`
**Type:** int, **frames**
**Stage:** counter
**Code:** [`run_yolo11_tracking.py:393`](../run_yolo11_tracking.py:393) — `confirmed = hits >= self.min_hits`. Tracks with fewer hits don't have their line crossings counted, even if the line is crossed geometrically.

This is the **most powerful knob for fixing overcounting from ghost detections.** A flickering false detection that lives for 1–2 frames and crosses a line will *not* be counted at `min_hits=3`.

| Push UP (3 → 5) | Push DOWN (3 → 1) |
|---|---|
| Ghost detections die before counting | Fast vehicles that only appear in 1–2 frames get counted |
| Less false positives in counts | Risk: any flicker now counts |
| Risk: very fast vehicles miss the count window | — |

**Sensible range:** 1–8.
- High frame rate, slow scene → can push to 5–7 safely.
- Low frame rate (10 fps) or fast scene → keep at 1–3 or you'll miss real vehicles.

**This is the first knob to try when overcounting.**

---

## `buffer` — line-crossing proximity zone

**Default:** `0`
**Type:** int, **pixels**
**Stage:** counter
**Code:** [`run_yolo11_tracking.py:411`](../run_yolo11_tracking.py:411) — when set to 0, only exact line crossings (sign change of cross-product) count. When > 0, a track is also counted when its centroid enters a zone within `buffer` pixels of the line *from outside the zone*.

| `buffer = 0` | `buffer = 20` |
|---|---|
| Strict — must cross the line geometrically | Counts vehicles that pass *near* the line |
| Risk: tracks that disappear right at the line don't count | Risk: a vehicle stopping near the line could trigger multiple counts |

**Sensible range:** 0 (default, recommended) or 10–30 if you have specific issues. Most users leave this at 0. Bump up only when you see the diff is `-1` or `-2` and you've watched the annotated video and seen vehicles that *almost* crossed.

**Interacts with:** line geometry. If you draw the line in a place where tracker frequently loses tracks just before crossing (e.g. behind a pole), buffer compensates — but the better fix is moving the line.

---

## `deduplicate` — overlap suppression on/off

**Default:** `true`
**Type:** boolean
**Stage:** dedup
**Code:** [`run_yolo11_tracking.py:802`](../run_yolo11_tracking.py:802) — when true, runs `deduplicate_detections()` after raw YOLO output to suppress:
- Two detections of the **same class** with high overlap (NMS missed them).
- Two detections of **different but equivalent vehicle classes** (e.g. "car" + "truck" on one vehicle) — uses the `equiv_classes={car, motorcycle, bus, truck}` set.

| `true` (default) | `false` |
|---|---|
| Single vehicle = single detection, even if YOLO returns multiple | Each YOLO detection becomes its own track |
| Risk: rarely an issue — only flip OFF if you suspect it's hiding distinct objects | Risk: dramatic overcounting (one truck → "car + truck" → counted twice) |

**Recommendation:** keep `true`. The only reason to flip it off is debugging — to see what YOLO actually emits before suppression.

---

## Quick lookup table

| Param | Default | Unit | Range | Stage | First-resort for |
|---|---|---|---|---|---|
| `confidence` | 0.3 | prob | 0.2–0.5 | detector | flicker (UP), missed objects (DOWN) |
| `iou` | 0.45 | IoU | 0.4–0.5 | detector / dedup | usually leave alone |
| `min_iou` | 0.15 | IoU | 0.10–0.30 | tracker | ID swaps (UP) |
| `max_distance` | 200 | pixels | 100–300 | tracker | lost fast vehicles (UP) |
| `max_disappeared` | 50 | frames | 20–100 | tracker | broken tracks across occlusion (UP) |
| `min_hits` | 3 | frames | 1–8 | counter | overcounting from ghosts (UP) |
| `buffer` | 0 | pixels | 0–30 | counter | edge crossings missed (UP) |
| `deduplicate` | true | bool | — | dedup | leave true |

For the actual tuning loop see [HOWTO.md](HOWTO.md).
