import { createStickFigure } from "@zachshotamartin/stick-figure";

/** The same literal stick figure used by Hide and Seek, at physical landmarks. */
export function createDiver(onReady) {
  return createStickFigure({ color: "#348cdd", onReady });
}

export { NEUTRAL_POSE } from "@zachshotamartin/stick-figure";
