# Specific-dive practice and directional takeoff

This phase continues the existing 233-input shared motor policy. Visitors and
the practice sampler can request a legal dive. Joint actions are still learned
through the same MuJoCo physics; no trajectories are supplied. Autonomous
six-dive routines remain a separate test of the selector.

## Practice distribution

With `--goal-practice --direction-practice --motor-curriculum adaptive
--reward-mode phase-dense`, motor fundamentals initially receive 50% of
assignments, specific complete dives 30%, and autonomous dives 20%. As all
motor skills improve, fundamentals decline to 20%, specific targets grow to
48%, and autonomous assignments grow to 32%. These are assignment proportions,
not equal numbers of simulation frames.

Specific targets come from the existing legal mask for category, apparatus,
height and used dive numbers. Twenty percent of their sampling distribution
is uniform over legal targets; the remainder favors underpracticed targets,
recent learning progress, and difficulty near demonstrated skills. This
prevents the selector from permanently avoiding difficult targets. Targeted
practice trains motor actions, not declaration choices. Its scores are
excluded from autonomous results, qualification and champion selection.

Takeoff practice balances all four standing facing/direction combinations.
It measures world-Y angular momentum at final contact loss and signed actual
somersault progress. Wrong-direction takeoff cannot pass merely by jumping.
The bounded momentum term saturates at a modest 12 kg m²/s floor: more spin
does not collect more credit. Takeoff speed, rise and board collisions remain
separate costs. Directional feedback before release is potential shaping with
terminal cancellation. Competition difficulty and execution are unchanged.

Takeoff readiness must exceed its threshold in every direction before practice
duration grows from 1.4 to 4.8 seconds. Longer trials include declared posture
and entry feedback to connect launching, flight control and opening. Other
motor tasks retain their own progress.

## Continuing a checkpoint

`--continue-from` is an explicit learning-phase migration, not exact resume.
It requires the same model format, observation layout, actor/critic architecture,
physics asset, scoring rules and core training options. It copies every learned
network weight and Adam state and preserves cumulative steps, update count,
elapsed training time, prior evaluations and checkpoint lineage. Four initial
updates fit only the independent critic to the revised experience distribution;
the actor remains bit-for-bit unchanged during those updates.

Partial episodes and recovery snapshots are retained in the parent checkpoint,
but new episodes begin for the changed objective. Existing shape/aerial/entry
readiness transfers. Jump-only readiness resets because it never established
directional skill. A separate output directory preserves all old artifacts;
`continuation.json` identifies the exact parent hash and boundary. Subsequent
`--resume` restores the new curriculum, targets, optimizer, RNG and world state
exactly. The running Python process cannot import changed training code live.

## Evaluation and browser

Full-routine reports include correct/wrong signed rotation, signed takeoff
momentum, jump height, posture, entry and execution by category. Separate
`target-evaluations` test ten explicit targets from real 10 m board starts,
including paired disturbances. Assigned-target performance is never counted
as successful autonomous selection. Curriculum counters are not proof of
mastery; promotion still requires the full qualification suite.

The browser supports specific targets and Auto. Category filters the list;
the same legality function validates both the list and simulation request.
Specific targets run as repeatable practice; Auto runs nonrepeating rounds.
Displayed availability is not a claim that a development policy has mastered
each target. This interface update does not publish an unqualified model.
