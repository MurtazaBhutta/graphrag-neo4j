"""A seeded, synthetic field-service world: customers, sites, assets, engineers, parts and jobs.

Same seed and size, same world, byte for byte. That is what lets the benchmark compute gold
answers in plain Python and compare them with what Neo4j returns.

A few patterns are planted on purpose, so the benchmark has something real to find:

* **Repeat failures.** Some jobs are follow-ups of an earlier job on the same asset, and a handful
  of "problem assets" are revisited again and again.
* **Free-text themes.** Small sets of jobs whose engineer notes describe the same problem
  (water through a ceiling, a burning smell at a board, lift doors re-opening) in deliberately
  different words. No background note uses those phrasings, so the planted jobs are the gold
  answer for the semantic questions.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

START = date(2024, 1, 1)
DAYS = 731  # 2024-01-01 .. 2025-12-31

ASSET_TYPES = {
    # type: (trade, makes, faults)
    "boiler": (
        "gas",
        ["Ideal Evomax", "Hamworthy Wessex", "Potterton Sirius"],
        ["no heating", "no hot water", "boiler lockout", "pressure loss"],
    ),
    "chiller": (
        "hvac",
        ["Carrier AquaSnap", "Daikin EWAD", "Trane CGAM"],
        ["high temperature alarm", "compressor trip", "refrigerant leak"],
    ),
    "ahu": (
        "hvac",
        ["Fläkt eQ", "Swegon GOLD", "Systemair Topvex"],
        ["filter blocked", "fan failure", "poor airflow"],
    ),
    "lift": (
        "lifts",
        ["Otis Gen2", "Kone MonoSpace", "Schindler 3300"],
        ["lift out of service", "levelling fault", "alarm activation"],
    ),
    "fire_alarm": (
        "fire",
        ["Gent Vigilon", "Advanced MxPro", "Kentec Syncro"],
        ["false alarm", "panel fault", "detector fault"],
    ),
    "electrical_board": (
        "electrical",
        ["Schneider Acti9", "Hager Invicta", "Eaton xEnergy"],
        ["tripping circuit", "power loss", "RCD fault"],
    ),
    "generator": (
        "electrical",
        ["Cummins C150", "Perkins 1100", "FG Wilson P165"],
        ["failed to start", "battery fault", "service due"],
    ),
}
ASSET_WEIGHTS = {
    "boiler": 5,
    "chiller": 2,
    "ahu": 3,
    "lift": 2,
    "fire_alarm": 3,
    "electrical_board": 4,
    "generator": 1,
}

PARTS = [
    # sku, name, unit cost (pence), asset types it fits
    ("P-101", "Flame sensor", 3800, ("boiler",)),
    ("P-102", "Ignition electrode", 2400, ("boiler",)),
    ("P-103", "Diverter valve", 9600, ("boiler",)),
    ("P-104", "Pressure relief valve", 4500, ("boiler",)),
    ("P-105", "Circulating pump", 18500, ("boiler", "chiller")),
    ("P-201", "Compressor contactor", 6200, ("chiller",)),
    ("P-202", "Refrigerant R410A (kg)", 5200, ("chiller",)),
    ("P-203", "Expansion valve", 14200, ("chiller",)),
    ("P-301", "Panel filter G4", 1800, ("ahu",)),
    ("P-302", "Bag filter F7", 4200, ("ahu",)),
    ("P-303", "Fan belt", 2600, ("ahu",)),
    ("P-304", "EC fan motor", 46000, ("ahu",)),
    ("P-401", "Door operator belt", 7800, ("lift",)),
    ("P-402", "Landing door roller", 3100, ("lift",)),
    ("P-403", "Levelling sensor", 12400, ("lift",)),
    ("P-501", "Optical smoke detector", 3900, ("fire_alarm",)),
    ("P-502", "Heat detector", 3600, ("fire_alarm",)),
    ("P-503", "Panel standby battery", 2900, ("fire_alarm", "generator")),
    ("P-601", "MCB 32A", 900, ("electrical_board",)),
    ("P-602", "RCBO 20A", 3400, ("electrical_board",)),
    ("P-603", "Busbar shroud", 1500, ("electrical_board",)),
    ("P-701", "Starter battery 12V", 11500, ("generator",)),
    ("P-702", "Fuel filter", 2200, ("generator",)),
]

# Background note fragments. Nothing here may resemble the planted themes below.
SYMPTOMS = {
    "no heating": [
        "Radiators cold throughout building.",
        "No heat to east wing.",
        "Heating not coming on at scheduled time.",
    ],
    "no hot water": ["No hot water to washrooms.", "Hot water temperature below 40C."],
    "boiler lockout": ["Boiler showing lockout code.", "Boiler locked out overnight."],
    "pressure loss": ["System pressure dropping to 0.3 bar.", "Pressure gauge reading low."],
    "high temperature alarm": [
        "BMS high chilled water temperature alarm.",
        "Chilled water flow temperature above setpoint.",
    ],
    "compressor trip": ["Compressor 2 tripped on overload.", "Chiller tripped on HP fault."],
    "refrigerant leak": ["Low refrigerant charge alarm.", "Suspected refrigerant loss."],
    "filter blocked": ["Filter differential pressure alarm.", "Dirty filter alarm on BMS."],
    "fan failure": ["Supply fan not running.", "Extract fan failed alarm."],
    "poor airflow": ["Low airflow complaints in meeting rooms.", "Stuffy conditions reported."],
    "lift out of service": ["Lift not responding to calls.", "Lift stopped at floor 3."],
    "levelling fault": ["Car stopping 20mm below landing.", "Trip hazard at landing level."],
    "alarm activation": [
        "Lift alarm button activated, no persons trapped.",
        "Auto-dialler test failed.",
    ],
    "false alarm": ["Fire alarm activated with no fire.", "Unwanted alarm from zone 4."],
    "panel fault": ["Fire panel showing fault light.", "Panel reporting earth fault."],
    "detector fault": ["Detector in zone 2 reporting fault.", "Detector head missing."],
    "tripping circuit": ["Lighting circuit tripping.", "Sockets circuit keeps tripping."],
    "power loss": ["Loss of power to second floor.", "Partial power outage."],
    "RCD fault": ["RCD will not reset.", "RCD tripping intermittently."],
    "failed to start": [
        "Generator failed to start on weekly test.",
        "No start on mains fail test.",
    ],
    "battery fault": ["Generator battery voltage low.", "Battery charger alarm."],
    "service due": ["Annual service visit.", "Scheduled maintenance visit."],
}
ACTIONS = {
    "boiler": [
        "Cleaned burner and flame sensor, reset and tested.",
        "Replaced ignition electrode, combustion analysis within limits.",
        "Repressurised system and checked expansion vessel charge.",
        "Replaced diverter valve, tested hot water and heating.",
    ],
    "chiller": [
        "Reset compressor, checked operating pressures.",
        "Located and repaired leak, recharged refrigerant.",
        "Replaced compressor contactor, ran unit under load.",
    ],
    "ahu": [
        "Replaced filters and reset alarm.",
        "Replaced fan belt and tensioned.",
        "Checked dampers and rebalanced airflow.",
    ],
    "lift": [
        "Reset controller and ran test journeys.",
        "Adjusted levelling sensor.",
        "Checked auto-dialler and alarm line.",
    ],
    "fire_alarm": [
        "Cleaned detector and tested zone.",
        "Replaced faulty detector head.",
        "Traced and cleared earth fault on loop.",
    ],
    "electrical_board": [
        "Traced fault to faulty appliance, removed from circuit.",
        "Replaced MCB and tested circuit.",
        "Replaced RCBO, insulation test OK.",
    ],
    "generator": [
        "Replaced starter battery, test run OK.",
        "Changed fuel filter, ran on load.",
        "Completed service and load bank test.",
    ],
}
OUTCOMES = [
    "Left operational.",
    "Customer informed.",
    "Monitoring advised.",
    "Signed off by site contact.",
    "Further visit not required.",
]

THEMES = {
    "ceiling_leak": (
        ("boiler", "ahu", "chiller"),
        [
            "Water coming through the ceiling tiles above reception; traced to a weeping joint "
            "in the void.",
            "Tenant reported drips landing on desks from the suspended ceiling; condensate tray "
            "overflowing above the office.",
            "Damp patch spreading across the corridor ceiling; stained tiles removed and failed "
            "fitting replaced.",
            "Part of the ceiling came down after water ingress from plant above; isolated and "
            "made safe.",
            "Leak through a light fitting in the first floor ceiling, source was the drain line.",
            "Staff put buckets under a leak from the ceiling void; split pipe repaired and area "
            "dried out.",
        ],
    ),
    "burning_smell": (
        ("electrical_board",),
        [
            "Strong burning smell near the distribution board; scorch marks on busbar, damaged "
            "breaker replaced.",
            "Occupants reported a smell of melting plastic in the riser cupboard; loose neutral "
            "had overheated.",
            "Smoke seen from the consumer unit; charred cable insulation found on circuit 7.",
            "Hot electrical odour in plant room traced to an arcing contactor.",
            "Discoloured terminals and an acrid smell at the sub-main panel; terminations remade.",
            "Burnt smell reported in the kitchen; overheating isolator found and replaced.",
        ],
    ),
    "lift_doors": (
        ("lift",),
        [
            "Lift doors keep re-opening before they close; safety edge misaligned.",
            "Passengers waiting ages because the doors would not shut fully; light curtain dirty.",
            "Car doors bouncing back open repeatedly on level 2; door operator adjusted.",
            "Doors reverse mid-close with nothing in the way; door detector replaced.",
            "Doors cycle open and shut again and again before the car will move; recalibrated.",
            "Intermittent door re-open loop causing long waits; cleaned and realigned sensors.",
        ],
    ),
}

PREFIXES = [
    "Harbour",
    "Northgate",
    "Elmwood",
    "Kingsgate",
    "Riverside",
    "Oakfield",
    "Westbrook",
    "Ashford",
    "Beacon",
    "Crown",
    "Summit",
    "Meridian",
    "Pinnacle",
    "Granite",
    "Lakeside",
    "Highfield",
    "Fairview",
    "Stonebridge",
    "Redwood",
    "Silverline",
]
SUFFIXES = [
    ("Group", "office"),
    ("Retail", "retail"),
    ("Estates", "property"),
    ("Healthcare", "healthcare"),
    ("Schools Trust", "education"),
    ("Hotels", "hospitality"),
]
CITIES = [
    ("Bristol", "BS"),
    ("Leeds", "LS"),
    ("Manchester", "M"),
    ("London", "EC"),
    ("Birmingham", "B"),
    ("Glasgow", "G"),
    ("Cardiff", "CF"),
    ("Nottingham", "NG"),
    ("Liverpool", "L"),
    ("Sheffield", "S"),
    ("Newcastle", "NE"),
    ("Reading", "RG"),
]
SITE_KINDS = {
    "office": ["House", "Plaza", "Tower", "Court"],
    "retail": ["Store", "Retail Park", "Superstore"],
    "property": ["Business Park", "Exchange", "Quarter"],
    "healthcare": ["Clinic", "Medical Centre", "Care Home"],
    "education": ["Academy", "Primary School", "College"],
    "hospitality": ["Hotel", "Lodge", "Conference Centre"],
}
FIRST = [
    "Sam",
    "Alex",
    "Priya",
    "Tom",
    "Aisha",
    "Daniel",
    "Chloe",
    "Imran",
    "Grace",
    "Liam",
    "Fatima",
    "Owen",
    "Zara",
    "Ben",
    "Hannah",
    "Kofi",
    "Ella",
    "Ravi",
    "Megan",
    "Jamal",
    "Sophie",
    "Callum",
    "Noor",
    "Ethan",
]
LAST = [
    "Patel",
    "Hughes",
    "Khan",
    "Morgan",
    "Walsh",
    "Okafor",
    "Davies",
    "Ahmed",
    "Taylor",
    "Fraser",
    "Begum",
    "Lewis",
    "Murray",
    "Shah",
    "Evans",
    "Clarke",
    "Rahman",
    "Wright",
]
REGIONS = ["South West", "North", "North West", "London", "Midlands", "Scotland", "Wales"]


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    sector: str


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    city: str
    postcode: str
    customer_id: str


@dataclass(frozen=True)
class Asset:
    id: str
    type: str
    make: str
    site_id: str


@dataclass(frozen=True)
class Engineer:
    id: str
    name: str
    trade: str
    region: str


@dataclass(frozen=True)
class Part:
    sku: str
    name: str
    unit_cost_pence: int


@dataclass(frozen=True)
class Job:
    id: str
    reported_on: str  # ISO date
    site_id: str
    asset_id: str
    engineer_id: str
    fault_category: str
    priority: str  # emergency | urgent | routine
    status: str  # completed | open | cancelled
    sla_met: bool
    cost_pence: int
    notes: str
    parts: tuple[tuple[str, int], ...] = ()
    follow_up_of: str | None = None


@dataclass
class World:
    seed: int
    customers: dict[str, Customer] = field(default_factory=dict)
    sites: dict[str, Site] = field(default_factory=dict)
    assets: dict[str, Asset] = field(default_factory=dict)
    engineers: dict[str, Engineer] = field(default_factory=dict)
    parts: dict[str, Part] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    themes: dict[str, list[str]] = field(default_factory=dict)  # theme -> planted job ids
    focus_customer: str = ""  # the customer the filtered semantic question is about

    def customer_of(self, job: Job) -> Customer:
        return self.customers[self.sites[job.site_id].customer_id]

    def fingerprint(self) -> str:
        """Changes if anything about the generated data changes. Stored in Neo4j on load."""
        digest = hashlib.sha256()
        for job in self.jobs.values():
            digest.update(repr(job).encode())
        return digest.hexdigest()[:16]


def generate(n_jobs: int = 50_000, seed: int = 7) -> World:
    rng = random.Random(seed)
    world = World(seed=seed)
    seq = itertools.count(1)

    names = rng.sample([(p, s) for p in PREFIXES for s in SUFFIXES], 40)
    for i, (prefix, (suffix, sector)) in enumerate(names, start=1):
        world.customers[f"C{i:03d}"] = Customer(f"C{i:03d}", f"{prefix} {suffix}", sector)

    site_names: set[str] = set()
    for customer in world.customers.values():
        prefix = customer.name.split()[0]
        for _ in range(rng.randint(4, 16)):
            city, area = rng.choice(CITIES)
            name = f"{prefix} {city} {rng.choice(SITE_KINDS[customer.sector])}"
            n = 2
            while name in site_names:
                name, n = f"{name.rsplit(' No.', 1)[0]} No.{n}", n + 1
            site_names.add(name)
            sid = f"S{len(world.sites) + 1:04d}"
            letters = "ABDEFGHJLNPQRSTUWXYZ"
            postcode = (
                f"{area}{rng.randint(1, 20)} {rng.randint(1, 9)}"
                f"{rng.choice(letters)}{rng.choice(letters)}"
            )
            world.sites[sid] = Site(sid, name, city, postcode, customer.id)

    types, weights = zip(*ASSET_WEIGHTS.items(), strict=True)
    for site in world.sites.values():
        for asset_type in dict.fromkeys(rng.choices(types, weights, k=rng.randint(2, 6))):
            aid = f"A{len(world.assets) + 1:05d}"
            world.assets[aid] = Asset(
                aid, asset_type, rng.choice(ASSET_TYPES[asset_type][1]), site.id
            )

    engineer_names = rng.sample([f"{f} {last}" for f in FIRST for last in LAST], 120)
    trades = sorted({t for t, _, _ in ASSET_TYPES.values()})
    for i, name in enumerate(engineer_names, start=1):
        world.engineers[f"E{i:03d}"] = Engineer(
            f"E{i:03d}", name, trades[i % len(trades)], rng.choice(REGIONS)
        )
    by_trade: dict[str, list[str]] = {}
    for e in world.engineers.values():
        by_trade.setdefault(e.trade, []).append(e.id)
    # Each engineer has a first-time-fix "skill" that drives SLA outcomes.
    skill = {eid: rng.uniform(0.55, 0.97) for eid in world.engineers}

    for sku, name, cost, _ in PARTS:
        world.parts[sku] = Part(sku, name, cost)
    parts_for = {t: [p[0] for p in PARTS if t in p[3]] for t in ASSET_TYPES}

    # Busy sites get more work (a long tail, like a real portfolio).
    site_ids = list(world.sites)
    site_weight = [1 / (rank + 3) for rank in range(len(site_ids))]
    rng.shuffle(site_weight)
    assets_at: dict[str, list[str]] = {}
    for a in world.assets.values():
        assets_at.setdefault(a.site_id, []).append(a.id)

    drafts: list[dict] = []
    for _ in range(n_jobs):
        site_id = rng.choices(site_ids, site_weight)[0]
        asset = world.assets[rng.choice(assets_at[site_id])]
        drafts.append(
            _draft(rng, seq, asset, by_trade, skill, parts_for, world, day=rng.randrange(DAYS))
        )

    _add_follow_ups(rng, seq, drafts, by_trade, skill, parts_for, world)
    _plant_themes(rng, drafts, world)

    drafts.sort(key=lambda d: (d["day"], d["seq"]))
    ids = {d["seq"]: f"J-{i:06d}" for i, d in enumerate(drafts, start=1)}
    for d in drafts:
        job = Job(
            id=ids[d["seq"]],
            reported_on=(START + timedelta(days=d["day"])).isoformat(),
            site_id=d["site_id"],
            asset_id=d["asset_id"],
            engineer_id=d["engineer_id"],
            fault_category=d["fault"],
            priority=d["priority"],
            status=d["status"],
            sla_met=d["sla_met"],
            cost_pence=d["cost_pence"],
            notes=d["notes"],
            parts=d["parts"],
            follow_up_of=ids[d["follow_up_of"]] if d["follow_up_of"] is not None else None,
        )
        world.jobs[job.id] = job
    world.themes = {t: sorted(ids[s] for s in seqs) for t, seqs in world.themes.items()}
    return world


def _draft(rng, seq, asset, by_trade, skill, parts_for, world, *, day, fault=None) -> dict:
    trade, _, faults = ASSET_TYPES[asset.type]
    fault = fault or rng.choice(faults)
    engineer_id = rng.choice(by_trade[trade])
    priority = rng.choices(["emergency", "urgent", "routine"], [15, 25, 60])[0]
    status = rng.choices(["completed", "open", "cancelled"], [90, 6, 4])[0]
    parts: tuple[tuple[str, int], ...] = ()
    if status == "completed" and rng.random() < 0.55:
        chosen = rng.sample(
            parts_for[asset.type], k=min(len(parts_for[asset.type]), rng.randint(1, 2))
        )
        parts = tuple(sorted((sku, rng.randint(1, 3)) for sku in chosen))
    labour = {"emergency": 18000, "urgent": 12500, "routine": 8500}[priority]
    cost = (
        0
        if status == "cancelled"
        else labour + sum(world.parts[s].unit_cost_pence * q for s, q in parts)
    )
    sla_met = status != "open" and rng.random() < skill[engineer_id] + (
        0.05 if priority == "routine" else -0.05
    )
    notes = " ".join(
        [rng.choice(SYMPTOMS[fault]), rng.choice(ACTIONS[asset.type]), rng.choice(OUTCOMES)]
    )
    return {
        "seq": next(seq),
        "day": day,
        "site_id": asset.site_id,
        "asset_id": asset.id,
        "engineer_id": engineer_id,
        "fault": fault,
        "priority": priority,
        "status": status,
        "sla_met": sla_met,
        "cost_pence": cost,
        "notes": notes,
        "parts": parts,
        "follow_up_of": None,
    }


def _add_follow_ups(rng, seq, drafts, by_trade, skill, parts_for, world) -> None:
    """About 6% of jobs get a return visit within 30 days; a few assets get a chain of them."""
    originals = rng.sample(drafts, k=len(drafts) * 6 // 100)
    problem = {d["seq"] for d in rng.sample(originals, k=min(8, len(originals)))}
    for original in originals:
        chain = rng.randint(3, 5) if original["seq"] in problem else 1
        previous = original
        for _ in range(chain):
            if previous["day"] + 1 >= DAYS:
                break
            day = min(DAYS - 1, previous["day"] + rng.randint(1, 30))
            asset = world.assets[previous["asset_id"]]
            visit = _draft(
                rng, seq, asset, by_trade, skill, parts_for, world, day=day, fault=previous["fault"]
            )
            visit["follow_up_of"] = previous["seq"]
            drafts.append(visit)
            previous = visit


def _plant_themes(rng, drafts, world) -> None:
    """Rewrite the notes of a few jobs per theme. Ceiling leaks are common across customers
    (24 jobs, 3 of them at one chosen customer in 2025); the other two themes have 6 each,
    split evenly across 2024 and 2025."""
    world.themes = {}
    used: set[int] = set()

    def pick(asset_types, *, customer=None, year=None, k):
        pool = [
            d
            for d in drafts
            if d["seq"] not in used
            and d["follow_up_of"] is None
            and d["status"] == "completed"
            and world.assets[d["asset_id"]].type in asset_types
            and (customer is None or world.sites[d["site_id"]].customer_id == customer)
            and (year is None or (START + timedelta(days=d["day"])).year == year)
        ]
        chosen = rng.sample(pool, k=k)
        used.update(d["seq"] for d in chosen)
        return chosen

    leak_types, leak_notes = THEMES["ceiling_leak"]
    focus = world.focus_customer = _busiest_customer(drafts, world)
    leak_jobs = pick(leak_types, customer=focus, year=2025, k=3) + pick(leak_types, k=21)
    for i, d in enumerate(leak_jobs):
        d["notes"] = leak_notes[i % len(leak_notes)]
    world.themes["ceiling_leak"] = [d["seq"] for d in leak_jobs]

    for theme in ("burning_smell", "lift_doors"):
        asset_types, notes = THEMES[theme]
        chosen = pick(asset_types, year=2024, k=3) + pick(asset_types, year=2025, k=3)
        for d, text in zip(chosen, notes, strict=True):
            d["notes"] = text
        world.themes[theme] = [d["seq"] for d in chosen]


def _busiest_customer(drafts, world) -> str:
    counts: dict[str, int] = {}
    for d in drafts:
        cid = world.sites[d["site_id"]].customer_id
        counts[cid] = counts.get(cid, 0) + 1
    return max(sorted(counts), key=counts.__getitem__)


def render_record(job: Job, world: World) -> str:
    """One job as text: what a typical RAG-over-records pipeline embeds. Every field is here, so
    the vector baseline is not starved of information; it lacks only structure."""
    site = world.sites[job.site_id]
    asset = world.assets[job.asset_id]
    parts = ", ".join(f"{world.parts[s].name} x{q}" for s, q in job.parts) or "none"
    lines = [
        f"Job {job.id} reported {job.reported_on}",
        f"Customer: {world.customers[site.customer_id].name}",
        f"Site: {site.name}, {site.city} {site.postcode}",
        f"Asset: {asset.type.replace('_', ' ')} {asset.make} ({asset.id})",
        f"Engineer: {world.engineers[job.engineer_id].name}",
        f"Fault: {job.fault_category} | Priority: {job.priority} | Status: {job.status} | "
        f"SLA met: {'yes' if job.sla_met else 'no'} | Cost: £{job.cost_pence / 100:.2f}",
        f"Parts: {parts}",
    ]
    if job.follow_up_of:
        lines.append(f"Follow-up of {job.follow_up_of}")
    lines.append(f"Notes: {job.notes}")
    return "\n".join(lines)
