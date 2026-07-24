We'll check that later. First, we need to debug another minor issue.

-------------
PC has LAN + proxy (required for internet): 172.30.10.10:3128
Proxy exclusions: localhost, 127.0.0.0/8, ::1, 127.0.0.1, 0.0.0.0

Robot is running teleimager server.
On PC, I try to access webrtc video of robot's head camera at this address: https://192.168.123.164:60001/. The feed is not visible and page gives error: ERR_TUNNEL_CONNECTION_FAILED

Meta Quest is connected to WiFi + No Proxy. When I visit https://192.168.123.164:60001/ page on Meta Quest. It does not open and gives error: ERR_TIMED_OUT.


When I run teleoperation and open vuer on PC at https://0.0.0.0:8012/?ws=wss://0.0.0.0:8012. Camera feed does not show here either. Similarly, when I open vuer on meta quest, camera feed does not show there too. Just blank. But teleoperation still works, hands, fingers and arms move just like they should when I move my hands.

--------------

When I add these proxy exclusions in PC: localhost, 127.0.0.0/8, ::1, 127.0.0.1, 0.0.0.0, 192.168.123.164, 192.168.123.161

Then Camera feed starts showing fine in the PC in vuer and at https://192.168.123.164:60001.
But the camera feed still does not show up when I open vuer in meta quest. Teleoperation of hands, fingers, arms still work fine.

When proxy is enabled on wifi (no exclusions) in meta quest, then vuer does not open at all. vuer & https://192.168.123.164:60001 gives same ERR_TIMED_OUT error.

The goal:
Make camera feed accessible in meta quest while also making sure arms and hands both are teleoperated successfully.