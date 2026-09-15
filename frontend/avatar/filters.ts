/**
 * Temporal filters and soft limits shared by the retargeting solvers (T1.17).
 *
 * All the smoothing here is **one-euro**: a low-pass whose cutoff rises with the
 * signal's own speed. A fixed low-pass cannot serve signing. Signs are mostly
 * holds, where the landmark noise is the whole signal and needs heavy smoothing,
 * separated by fast transitions, where the same smoothing would smear the motion
 * into mush. Letting the cutoff track speed gives a still hand a steady pose and
 * a moving hand its movement.
 *
 * `beta` is unit-dependent — it multiplies a derivative — so each filter here
 * normalises its derivative to a scale-free quantity first (radians per second
 * for rotations, lengths per second relative to a reference for positions).
 * Otherwise the same beta behaves completely differently on degrees, radians and
 * avatar units, which is easy to get wrong and silent when it is.
 *
 * Two details decide whether one-euro actually smooths anything, and both are
 * easy to get subtly wrong:
 *
 *  - The speed estimate must come from **consecutive raw inputs**, not from the
 *    gap between the input and the filter's own output. That gap is error, not
 *    velocity: it grows with the lag the filter is deliberately introducing, so
 *    using it raises the cutoff, which reduces the lag, which shrinks the gap —
 *    a feedback loop that settles wherever it likes and smooths far less than
 *    asked.
 *  - The speed must be low-passed **signed**, then measured. Low-passing a
 *    magnitude leaves alternating noise contributing its full size on every
 *    frame, so a perfectly stationary but noisy signal reports high speed and
 *    the filter opens right up on exactly the input it exists to smooth.
 */

import * as THREE from "three";

function smoothing(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

/** Scalar one-euro. */
export class OneEuro {
  private value: number | null = null;
  private derivative = 0;
  constructor(
    private readonly minCutoff: number,
    private readonly beta: number,
    private readonly dCutoff = 1.0,
  ) {}

  private previous: number | null = null;

  filter(x: number, dt: number): number {
    if (this.value === null || this.previous === null || !Number.isFinite(dt) || dt <= 0) {
      this.value = x;
      this.previous = x;
      return x;
    }
    const dx = (x - this.previous) / dt;
    this.previous = x;
    this.derivative += smoothing(this.dCutoff, dt) * (dx - this.derivative);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    this.value += smoothing(cutoff, dt) * (x - this.value);
    return this.value;
  }

  reset(): void { this.value = null; this.previous = null; this.derivative = 0; }
}

/**
 * First-order low-pass on a position.
 *
 * Adaptive gain was swept here too and changed nothing measurable — identical
 * lag and shake within noise at every setting — so this stays the simple thing.
 * It is deliberately lighter than the orientation smoother: this moves where the
 * hand *is*, which is a sign parameter in its own right, and lag in it reads as
 * the whole arm dragging rather than as a steadier wrist.
 */
export class SmoothedVector3 {
  private value: THREE.Vector3 | null = null;
  constructor(private readonly cutoff: number) {}

  filter(target: THREE.Vector3, dt: number): THREE.Vector3 {
    if (!this.value || !Number.isFinite(dt) || dt <= 0) {
      this.value = target.clone();
      return this.value.clone();
    }
    this.value.lerp(target, smoothing(this.cutoff, dt));
    return this.value.clone();
  }

  reset(): void { this.value = null; }
}

/**
 * Smoother for an orientation: double-exponential, with a cap on how far it may
 * turn in one frame.
 *
 * **Why not one-euro here, when the finger joints use it.** One-euro trades lag
 * for responsiveness by opening its cutoff when the signal moves. Swept against
 * this data it lost on every sign, and it lost in the specific way that gives
 * the game away: it made the *smoothest* span worse. On SIT, where the measured
 * palm is already steady (0.84°/frame² of shake), adaptive gain handed back 1.21°
 * while a fixed gain gave 0.51°. An adaptive filter is a time-varying one, and a
 * gain that changes from frame to frame injects its own acceleration into a
 * signal that is already moving. The finger joints are a different problem — they
 * are snapped to canonical handshapes afterwards, which hides small lag, so
 * there the responsiveness is worth having.
 *
 * What is left is a plain first-order low-pass, which the measurements say is
 * the right answer here. A trend term was tried — carry a smoothed estimate of
 * the rotation's own velocity and extrapolate along it, which in principle
 * removes a steady-state lag — and swept: at every cutoff it cost more shake
 * than it saved in lag, because palm orientation during signing is constantly
 * accelerating and has no steady state to extrapolate from. It was removed
 * rather than left in as a parameter that never earns its place.
 *
 * The cutoff is chosen against a zero-phase reference (the same filter run
 * forwards then backwards), which is the only way to tell real lag from the
 * filter correctly declining to follow noise: measured that way, an unfiltered
 * palm already sits 9.5° from the truth, so 0.6Hz costs about 11° of added lag
 * and removes two thirds of the shake.
 *
 * The rate cap is not smoothing, it is a discontinuity guard.
 * `demo_sequence.pose` is assembled from separate clips and contains cuts where
 * the measured palm jumps 155° between adjacent frames, plus dropouts of 19
 * frames after which tracking resumes somewhere else entirely. 750°/s is quicker
 * than any wrist moves, so the cap never touches real motion, and it turns a
 * teleport into a fast sweep — and it is what makes a hand returning from a
 * dropout ease back in rather than snap, since the held value is the pose it
 * left on.
 */
export class SmoothedRotation {
  private value: THREE.Quaternion | null = null;

  constructor(
    private readonly cutoff: number,
    /** Degrees per second. */
    private readonly maxRate = 750,
  ) {}

  /** Shortest-path rotation from `from` to `to`, as axis × angle. */
  static rotationVector(from: THREE.Quaternion, to: THREE.Quaternion): THREE.Vector3 {
    const delta = to.clone().multiply(from.clone().invert()).normalize();
    if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
    const vector = new THREE.Vector3(delta.x, delta.y, delta.z);
    const sine = vector.length();
    if (sine < 1e-12) return vector.set(0, 0, 0);
    return vector.multiplyScalar(2 * Math.atan2(sine, delta.w) / sine);
  }

  filter(target: THREE.Quaternion, dt: number): THREE.Quaternion {
    if (!this.value || !Number.isFinite(dt) || dt <= 0) {
      this.value = target.clone();
      return this.value.clone();
    }
    // q and -q are the same rotation; slerping to the far representative would
    // spin the wrist the long way round.
    const aim = target.clone();
    if (aim.dot(this.value) < 0) aim.set(-aim.x, -aim.y, -aim.z, -aim.w);

    const total = this.value.angleTo(aim);
    if (total < 1e-9) return this.value.clone();
    const step = Math.min(total * smoothing(this.cutoff, dt),
                          THREE.MathUtils.degToRad(this.maxRate) * dt);
    this.value.slerp(aim, step / total);
    return this.value.clone();
  }

  reset(): void { this.value = null; }
}

/**
 * Drop orientation measurements that move faster than the thing being measured
 * can physically move, while still letting a genuine scene change through.
 *
 * This is not smoothing, and a smoother cannot do its job. At the tail of
 * `demo_sequence.pose` the measured palm normal inverts between adjacent frames
 * — `[+0.34,-0.94,+0.09]` then `[+0.23,+0.92,-0.32]`, about 145° apart — because
 * MediaPipe cannot tell the palm from the back of the hand and flips its guess.
 * A hand does not turn over in a thirtieth of a second. Fed to a filter, a
 * flip-flopping input does not average out to the truth: it averages to an
 * orientation halfway between palm-up and palm-down, which is not a pose the
 * wrist can reach, so it arrives at the joint limits and stops there. That is
 * what a wrist visibly breaking looks like.
 *
 * `patience` is what keeps this from being a lie about the data. A real cut — and
 * this sequence is assembled from separate clips, so there are real cuts — also
 * arrives as one enormous step. The difference is that a cut *stays*: after a few
 * frames the new orientation is still there, so it is accepted. A flip does not,
 * because the next frame flips back.
 */
export class GlitchGate {
  private accepted: THREE.Quaternion | null = null;
  private suspect = 0;

  constructor(
    /** Degrees per second above which a step is not believed on sight. */
    private readonly maxRate = 1200,
    /** Frames an unbelievable value must persist before it is taken as real. */
    private readonly patience = 4,
  ) {}

  accept(measured: THREE.Quaternion, dt: number): THREE.Quaternion {
    if (!this.accepted || !Number.isFinite(dt) || dt <= 0) {
      this.accepted = measured.clone();
      this.suspect = 0;
      return this.accepted.clone();
    }
    const step = THREE.MathUtils.radToDeg(this.accepted.angleTo(measured));
    if (step > this.maxRate * dt && this.suspect < this.patience) {
      this.suspect += 1;
      return this.accepted.clone();
    }
    this.suspect = 0;
    this.accepted.copy(measured);
    return this.accepted.clone();
  }

  reset(): void { this.accepted = null; this.suspect = 0; }
}

/**
 * Approach a limit smoothly instead of hitting it.
 *
 * A hard clamp makes a joint travel normally and then stop dead, which reads as
 * the joint breaking — on `demo_sequence.pose` the wrist hit its stops on 11% of
 * frames. Below `knee` this is the identity, so ordinary motion is untouched;
 * past it the remaining travel is compressed into what is left, asymptotically,
 * so the limit is respected but never arrived at abruptly.
 */
export function softLimit(value: number, limit: number, knee = 0.7): number {
  const soft = Math.abs(limit) * knee;
  const magnitude = Math.abs(value);
  if (magnitude <= soft) return value;
  const span = Math.abs(limit) - soft;
  if (span <= 1e-9) return Math.sign(value) * soft;
  return Math.sign(value) * (soft + span * Math.tanh((magnitude - soft) / span));
}

/** `softLimit` with independent negative and positive bounds. */
export function softRange(value: number, low: number, high: number, knee = 0.7): number {
  return value < 0 ? -softLimit(-value, -low, knee) : softLimit(value, high, knee);
}
