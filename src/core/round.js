import { SKILLS } from "./control.js";
/** Practice scheduling is separate from learned motor control. Numeric codes cannot repeat. */
export class DiveRound {
  constructor(height = 7.5, seed = 1701) {
    this.number = 1;
    this.reset(height, seed);
  }
  reset(height = this.height, seed = this.seed) {
    this.height = height;
    this.seed = seed >>> 0;
    this.attempts = [];
    this.current = null;
    return this;
  }
  eligible() {
    return SKILLS.map((skill, index) => ({ ...skill, index })).filter(
      (s) => s.minHeight <= this.height,
    );
  }
  remaining() {
    const used = new Set(this.attempts.map((a) => a.code));
    return this.eligible().filter((s) => !used.has(s.code));
  }
  choose(preferred = null) {
    const remaining = this.remaining();
    if (!remaining.length) return null;
    let choice = remaining.find((s) => s.index === preferred);
    if (!choice) {
      this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
      const last = this.attempts.at(-1);
      const different = remaining.filter((s) => s.back !== last?.back);
      const pool = different.length ? different : remaining;
      choice = pool[this.seed % pool.length];
    }
    this.current = choice.index;
    return this.current;
  }
  record(result, skillIndex = this.current) {
    if (this.current === null || skillIndex !== this.current) return;
    const skill = SKILLS[this.current];
    if (this.attempts.some((a) => a.code === skill.code)) return;
    this.attempts.push({
      id: skill.id,
      code: skill.code,
      back: skill.back,
      execution: result.execution,
      points: result.total,
      valid: result.valid,
    });
  }
  nextRound() {
    this.number++;
    return this.reset();
  }
}
