# Teleoperation TODOs

## Low FPS / high latency of head-camera feed in Meta Quest

**Status:** open — revisit later. Feed works correctly in both PC and Quest Vuer as of 2026-07-14.

**Symptom:** camera feed renders in the Quest but at low frame rate / noticeable latency.

**Current path (the reason it's slow):** we run the ZMQ relay, not WebRTC.
- Robot (`cam_config_server.yaml`): `enable_zmq: true`, `enable_webrtc: false`.
- PC `ImageClient` pulls frames over ZMQ from `192.168.123.164:55555` via `robot0`.
- televuer ships each frame to the headset as `ImageBackground(format="jpeg", quality=80)`
  over the 8012 websocket (see `televuer/televuer.py` `main_image_*_zmq`).
- So the Quest receives JPEG-over-websocket, re-encoded and re-sent per frame, single
  8012 socket. That is inherently higher latency / lower throughput than WebRTC's H.264.

**Why we're on this path:** the Quest cannot reach the robot's internal `192.168.123.0/24`
network (L2 island, not routed). WebRTC's `webrtc_url` pointed the headset directly at
`192.168.123.164:60001`, which is unroutable from WiFi. See `NETWORK_SETUP.md`.

**Ideas to try (rough, unvalidated):**
1. Tune the ZMQ/JPEG path cheaply first:
   - Lower `ImageBackground` `quality` (80 → 50–60) and/or downscale before send.
   - Confirm `display_fps` in televuer isn't capping below camera fps; check the
     `asyncio.sleep(1.0/self.display_fps)` in the zmq render loop.
   - Check the 30fps camera vs teleop `--frequency 30` interaction (issue #172 referenced
     in teleop_hand_and_arm.py).
2. Real fix — get WebRTC working to the Quest via a *routable* robot address:
   - Robot is also on WiFi at `10.25.33.53`, which the Quest CAN route to.
   - Requires decoupling the WebRTC signaling host from the ZMQ host: keep ZMQ on wired
     `192.168.123.164`, point only `webrtc_url` at `10.25.33.53:60001`.
     `teleop_hand_and_arm.py:135` currently derives both from `--img-server-ip`.
   - Open questions before this is worth it (need to read teleimager WebRTC server):
     - What does the WebRTC signaling server bind to? Is it listening on the WiFi iface?
     - ICE candidates: does it only advertise `192.168.123.x` host candidates? If so the
       Quest completes signaling but media never flows — the make-or-break question.
     - Self-signed cert at `:60001` — must be accepted once in the Quest browser or the
       `/offer` fetch fails silently from inside the Vuer page.
   - Media still traverses the WiFi router, eating some of the latency win.

**Recommendation:** try (1) first — it's config-only and might be good enough. Only invest
in (2) if latency still bothers us.
