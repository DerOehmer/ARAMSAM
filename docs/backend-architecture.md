# Backend architecture

The application still starts with `python -m aramsam_annotator.main`. The visible
Qt controls, shortcuts, four image views, and existing public `App`, `Annotator`,
SAM, and worker entry points are retained. `configs.yaml` and existing dataset
layouts remain supported.

## Ownership and dependencies

| Module | Responsibility |
| --- | --- |
| `app.py` | Creates Qt, connects signals, owns the worker pool and model lifecycle; forwards existing callback methods. |
| `controllers/navigation.py` | Coordinates image selection, completed-image skipping, save dialogs, and loading annotations. |
| `controllers/inference.py` | Schedules embedding, proposals, and propagation; activates matching results. |
| `controllers/interaction.py` | Converts the existing button/mouse actions into annotation operations. |
| `controllers/experiments.py` | Preserves the structured/polygon experiment sequence and tutorial hooks. |
| `controllers/presentation.py` | Updates the four existing views and their selection/centering. |
| `annotator.py` | Owns the active annotation pair, editing state, mask IDs, and model instances; preserves the public annotation API. |
| `backend/annotations.py` | Annotation/mask records, image reading, and embedding data. Historical imports from `mask_visualizations` still work. |
| `backend/session.py` | Full-path image identity, transactional queue expansion, and preparation of matching current/next records. |
| `backend/editing.py` | Decisions, drawing, interactive prompts, deletion, and undo. |
| `backend/rendering.py` | Builds visualization data using the existing rendering algorithms. |
| `backend/storage.py` | Native and YOLO serialization, reload, format detection, and replacement of saved edits. No Qt or model dependency. |
| `backend/tasks.py` | Owns Qt receivers, delivers callbacks on the GUI thread, rejects stale results, and drains work on transitions. |
| `workers.py` | Model work with a shared exception-safe lifecycle and explicit mutex ownership. |
| `run_sam.py`, `run_yolo.py`, `tracker.py` | Existing model adapters and tracking algorithms. |

Controllers have an explicit reference to the application, and the editor and
renderer have an explicit reference to the annotator. These references do not
copy state. Services never locate dependencies through globals or dynamic
attribute forwarding. Public forwarding methods are intentional compatibility
points, including for existing tests and downstream callers.

## Image and task lifecycle

1. Finish outstanding work and deliver its queued callbacks before changing an
   image or model. Follow-up work submitted by callbacks is drained too.
2. Save or prompt for the outgoing annotation using the existing settings.
3. Expand the queue in a temporary list. A tiling failure leaves the live queue
   unchanged. Completed folders are skipped iteratively, preserving the original
   last-in-first-out image order.
4. Prepare both annotation records before committing a transition. Prefetched
   records and their embeddings are reused only when the full image path matches.
5. Embed the current image, then prefetch the next SAM1 image if configured.
   SAM2 rebuilds its image-pair inference state. Existing completed predecessors
   are reloaded before propagation into an unfinished image.
6. Deliver results only while their originating annotation/model still belongs
   to the active session. Prefetch completion does not reset the annotation timer.

SAM1 background encoding does not mutate the active predictor and therefore does
not hold the interactive-prediction mutex. Other model workers release their
locks in `finally` blocks before emitting results/errors. Hover/click inference
uses the same mutex. Failures are reported through the existing message-box UI,
including errors raised inside UI callbacks, rather than escaping a Qt signal.

Navigation still waits for outstanding model work at transition boundaries;
this change does not introduce cancellation of running inference. The previous
worker APIs and completion payloads remain supported. The misspelled public
`YoloPredicitonWorker` name remains available alongside `YoloPredictionWorker`.

## Saved annotations

Native export keeps `<image>_annots`, `img.jpg`, `annotations.jpg`, `masks/`, and
`log.json`. An additional `annotations.json` sidecar preserves class, origin,
color, timestamp, and bounding-box metadata. Old mask-only exports can still be
loaded. Loaded objects receive fresh session IDs.

Native output is fully staged before replacing the old directory. Resaving now
writes changed masks and removes deleted masks; failed encoding leaves the old
export intact. This is protection against ordinary write failures, not a
multi-file filesystem transaction or a power-loss durability guarantee.

YOLO keeps `images/`, `labels/`, and `control_images/`, with six-decimal normalized
coordinates. A completed YOLO export requires both its image and label file.
Both segmentation polygons and bounding boxes can be reloaded. Polygon export
retains the original largest-contour behavior, so it does not preserve holes or
additional disconnected components exactly. Native export preserves exact masks.

## Verification and extension

Run the existing environment's suite without a display:

```sh
QT_QPA_PLATFORM=offscreen samvenv/bin/python -m unittest discover -s tests
```

The suite covers legacy undo/device/embedding behavior, native and YOLO reload,
resaving/deletion and failed writes, path identity, tiling failures, completed
folders, cancelled dialogs, worker errors and thread affinity, stale results,
model reuse, experiment UI construction, and a real Qt polygon/save/reload flow.
Model weights are not needed for the automated suite; model construction and
inference are mocked where appropriate, with a tiny real torch encoder used in
the SAM1 predictor tests. Full-model GPU accuracy/performance and visual desktop
inspection require separate checks on suitable hardware.

Add an output format in `AnnotationRepository`; add an editing operation in
`AnnotationEditor`; add a UI workflow in its controller. Keep model-specific
behavior in the model adapters. New workers should extend `InferenceWorker` and
be submitted through `TaskDispatcher`, not connect directly to UI callbacks.
