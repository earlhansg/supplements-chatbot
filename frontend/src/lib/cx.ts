/** Tiny classnames joiner — avoids pulling in `clsx` for a handful of call sites. */
export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
