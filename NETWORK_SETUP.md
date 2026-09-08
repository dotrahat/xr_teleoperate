# Network Setup — Physical Teleoperation (Unitree G1 + Meta Quest)

Working configuration for running `teleop_hand_and_arm.py` against the physical robot.
Verified working: teleoperation ran successfully with hand tracking on `G1_23` + BrainCo hands.

---

## Topology

The robot and the PC share **one physical LAN segment**. The Quest is WiFi-only, and the
site network routes between LAN and WiFi, so the Quest reaches the PC over its LAN address.
The PC needs **no WiFi interface** of its own.

```
                    ┌──────────────────────────────┐
                    │  Meta Quest  (WiFi only)      │
                    │  10.25.33.x                   │
                    └───────────────┬───────────────┘
                                    │  WiFi
                          (site routes WiFi <-> LAN)
                                    │
   ┌────────────────────────────────┴──────────────────────────────┐
   │                        LAN switch                              │
   └───┬────────────────────────────────────────────┬──────────────┘
       │                                            │
  enp129s0  172.30.37.39/24                    Unitree G1
  robot0    192.168.123.222/24  (macvlan)      ├─ PC2          192.168.123.164
  ── same physical port ──                     └─ ctrl board   192.168.123.161
       │
   Teleop PC (Alienware-3)
```

Both robot addresses live on the **same L2 broadcast domain** as the PC — confirmed by
sub-millisecond ping RTT (~0.15–0.25 ms) and by real MAC addresses being learned directly
on the wire:

```
192.168.123.164  lladdr 4c:bb:47:a4:a2:d0
192.168.123.161  lladdr 7e:1d:75:60:f5:89
```

This L2 adjacency is **required**: CycloneDDS discovery (SPDP) is multicast, and multicast
does not survive a routed hop. Any design that reaches the robot through a router or a
tunnel will fail DDS discovery even if ICMP pings succeed.

---

## Address plan

| Host / interface | Address | Role |
|---|---|---|
| PC `enp129s0` | `172.30.37.39/24` (DHCP) | Default route; serves Vuer/WebXR to the Quest |
| PC `robot0` (macvlan on `enp129s0`) | `192.168.123.222/24`, no gateway | DDS to robot; camera ZMQ stream |
| Robot PC2 | `192.168.123.164` | Onboard PC; image server |
| Robot control board | `192.168.123.161` | Low-level motor control |
| Meta Quest | `10.25.33.x` (WiFi, DHCP) | XR client |

### Do NOT change PC2's IP address

`192.168.123.0/24` is the robot's **internal** network. PC2 talks to the low-level control
board at `192.168.123.161` over it. Moving PC2 onto the `172.30.37.0/24` subnet would break
its link to the motors. The robot's addressing is fixed; all adaptation happens on the PC
side. The rule is **add an address, never move one.**

---

## Why `robot0` (macvlan) exists — do not "simplify" this away

The obvious setup is to put both addresses on the single physical NIC:

```bash
# DO NOT DO THIS — it silently breaks DDS
ip addr add 192.168.123.222/24 dev enp129s0     # alongside 172.30.37.39/24
```

It pings fine, so it looks correct. It is not.

`unitree_sdk2py` passes CycloneDDS only an interface **name**, never an address — see
`unitree_sdk2py/core/channel_config.py`:

```xml
<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
```

With two IPv4 addresses on one NIC, Cyclone picks one itself, and it picks the wrong one:

```
interfaces: lo udp/127.0.0.1(q1) enp129s0 udp/172.30.37.39(q9) udp/192.168.123.222(q9)
selected interfaces: enp129s0 (index 2 priority 0)
ownip: udp/172.30.37.39            <-- LAN address, not the robot address
```

Cyclone would then advertise `172.30.37.39` as its DDS locator, and the robot has no route
back to that subnet. Discovery fails or behaves erratically.

There is no way to override this through the SDK: passing an IP instead of a name
(`--network-interface 192.168.123.222`) is rejected — Cyclone's `name=` attribute accepts
interface names only:

```
[ChannelFactory] create domain error. msg: Occurred upon initialisation of a cyclonedds.domain.Domain
```

**The fix is to give the robot subnet its own interface name with exactly one address.** A
macvlan sub-interface does this on the same physical wire — same L2 segment, full 1500 MTU,
zero ambiguity for Cyclone. Hence `robot0`, and hence `--network-interface=robot0`.

---

## Current runtime configuration (what is live now)

```
enp129s0         UP    172.30.37.39/24
robot0@enp129s0  UP    192.168.123.222/24     macvlan mode bridge, mtu 1500

default via 172.30.37.254 dev enp129s0 proto dhcp metric 20100
172.30.37.0/24  dev enp129s0 proto kernel scope link src 172.30.37.39 metric 100
192.168.123.0/24 dev robot0  proto kernel scope link src 192.168.123.222
```

Firewall: `ufw allow 8012` (Vuer / WebXR — already persistent, ufw rules survive reboot).

### How it was created (imperative — does NOT survive reboot)

```bash
sudo ip addr del 192.168.123.222/24 dev enp129s0        # undo the dual-IP attempt
sudo ip link add robot0 link enp129s0 type macvlan mode bridge
sudo ip addr add 192.168.123.222/24 dev robot0
sudo ip link set robot0 up
sudo ufw allow 8012
```

See "Persistence" below to make this permanent.

---

## Removed: the VXLAN tunnel

A previous attempt tunnelled the robot's L2 network over WiFi:

```
vxlan0: 192.168.123.222/24, vxlan id 123, remote 10.25.33.53, dev wlp130s0f0, mtu 1450
```

This has been **deleted** and must not come back. It was broken (its parent `wlp130s0f0`
was down, so it black-holed the entire `192.168.123.0/24` route into a dead tunnel — the
original symptom of "PC2 does not ping"), and it was unnecessary once the robot was on the
LAN switch. It also forced a 1450 MTU, which would fragment camera frames.

The robot's WiFi address (`10.25.33.53`) is **not used** for teleoperation. Do not point DDS
or the image server at it — it is a routed path and DDS multicast will not traverse it.

---

## Running teleoperation

```bash
conda activate tv-2
cd ~/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --input-mode=hand \
  --arm=G1_23 \
  --ee=brainco \
  --network-interface=robot0 \
  --img-server-ip=192.168.123.164 \
  --motion
```

For this G1-23 + BrainCo + hand-tracking combination, shoulder-relative arm scaling is
enabled by default using the measured `0.48 m` human shoulder-to-wrist length. The initial
head-relative shoulder anchors can be tuned without editing code:

```bash
--g1-23-left-shoulder-anchor=-0.1500072,0.10022,-0.15822 \
--g1-23-right-shoulder-anchor=-0.1500072,-0.10021,-0.15822
```

Use `--disable-g1-23-arm-scaling` for an unscaled A/B comparison. These options do not alter
the existing `0.15 m` forward and `0.45 m` vertical head-to-waist offsets.

Then on the Quest browser:

```
https://172.30.37.39:8012/?ws=wss://172.30.37.39:8012
```

The certs in `teleop/televuer` are self-signed; the Quest browser will show a warning that
must be accepted once. (Regenerating the cert with `172.30.37.39` in the SAN removes the
warning, if desired.)

Note: `--img-server-ip` defaults to `0.0.0.0` in the working tree, which suits simulation.
For the physical robot it **must** be passed explicitly as `192.168.123.164`.

---

## Persistence

`ip link` / `ip addr` changes are lost on reboot. NetworkManager currently reports `robot0`
as `connected (externally)` — meaning it is unmanaged and will vanish.

This host uses NetworkManager as its netplan renderer
(`/etc/netplan/01-network-manager-all.yaml`). **netplan has no macvlan support**, so the
persistent config belongs in a NetworkManager profile, which does support macvlan natively:

```bash
sudo ip link del robot0          # remove the manually-created one first

sudo nmcli connection add \
  type macvlan \
  con-name robot0 \
  ifname robot0 \
  dev enp129s0 \
  mode bridge \
  ipv4.method manual \
  ipv4.addresses 192.168.123.222/24 \
  ipv4.never-default yes \
  ipv4.link-local disabled \
  ipv6.method disabled \
  connection.autoconnect yes

sudo nmcli connection up robot0
```

`ipv4.never-default yes` is important: `robot0` must **not** install a default route — the
default must stay on `enp129s0` so the PC can still reach the Quest and the internet.
There is deliberately no gateway on the robot subnet.

Verify it survives:

```bash
sudo reboot
# after reboot:
ip -brief addr show robot0        # expect: UP  192.168.123.222/24
ip route | grep default           # expect: default via 172.30.37.254 dev enp129s0
```

---

## Verification checklist

```bash
# 1. Interfaces
ip -brief addr show enp129s0      # 172.30.37.39/24
ip -brief addr show robot0        # 192.168.123.222/24

# 2. Default route must be on the LAN NIC, not robot0
ip route | grep default

# 3. Robot reachable on the wire (expect sub-millisecond RTT)
ping -c3 192.168.123.164          # PC2
ping -c3 192.168.123.161          # control board

# 4. DDS binds the ROBOT address — the critical check
rm -f /tmp/cdds.LOG
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; \
           ChannelFactoryInitialize(0, networkInterface='robot0')"
grep ownip /tmp/cdds.LOG
#   expect:  ownip: udp/192.168.123.222
#   if it says 172.30.37.39, the macvlan is gone and DDS will not work
```

---

## Troubleshooting

**Pings work but DDS finds nothing.** Almost certainly the `ownip` problem. Check step 4
above. If `robot0` is missing, the config did not persist across a reboot.

**Quest cannot load the page.** Check `sudo ufw status` for 8012, and confirm the PC's LAN
address is still `172.30.37.39` (it is DHCP — if it changes, the Quest URL changes with it).
Consider a DHCP reservation for stability.

**`Address already assigned` when adding the IP.** It is already there; the command has
already been applied.

**Do not** try to reach the robot via its WiFi address, and do not re-create a VXLAN tunnel.
