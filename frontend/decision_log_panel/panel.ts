/**
 * Decision log panel — the UI half of T1.15 (architecture v3 §9).
 *
 * Renders the Observe/Decide/Action rows the backend emits, beside the avatar,
 * during the interaction. Two rules from §9 are structural here rather than
 * left to whoever calls it:
 *
 *  - The manual override control is rendered in **every** state (FR-16). It is
 *    part of the panel itself, not a separate widget that could be missing when
 *    it matters. §9 notes that visibly bounded autonomy reads as a strength.
 *  - Rows show structured decision factors only. The backend rejects any stage
 *    outside OBSERVE/DECIDE/ACTION, so raw model chain-of-thought cannot reach
 *    this panel even if something upstream tried to log it.
 */

export interface LogEntry {
  timestamp: number;
  stage: "OBSERVE" | "DECIDE" | "ACTION";
  segment_id: string;
  detail: string;
}

export interface LogPayload {
  override: "running" | "paused";
  entries: LogEntry[];
}

const STAGE_COLOUR: Record<LogEntry["stage"], string> = {
  OBSERVE: "#7aa2f7",
  DECIDE: "#e0af68",
  ACTION: "#9ece6a",
};

/** Highlight the words that carry the decision, so the panel is scannable. */
function emphasise(detail: string): string {
  const escaped = detail
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return escaped
    .replace(/\b(REFUSE|CLARIFY|TRANSLATE|HOLD|RELEASE)\b/g,
      '<span style="color:#f7768e;font-weight:600">$1</span>')
    .replace(/\b(lexicon_hit|language_backup|fingerspelling|unmatched)\b/g,
      '<span style="color:#bb9af7">$1</span>');
}

export class DecisionLogPanel {
  private readonly root: HTMLElement;
  private readonly list: HTMLElement;
  private readonly overrideButton: HTMLButtonElement;
  private readonly statusLabel: HTMLElement;
  private paused = false;
  private entries: LogEntry[] = [];

  onOverrideChange?: (paused: boolean) => void;

  constructor(root: HTMLElement) {
    this.root = root;
    this.root.innerHTML = "";
    this.root.style.cssText =
      "display:flex;flex-direction:column;height:100%;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c0caf5;background:#1a1b26;border-left:1px solid #2a2e3f";

    const header = document.createElement("div");
    header.style.cssText =
      "padding:10px 12px;border-bottom:1px solid #2a2e3f;display:flex;align-items:center;gap:10px;flex:0 0 auto";
    header.innerHTML =
      '<span style="font-weight:600;letter-spacing:.04em">DECISION LOG</span>' +
      '<span style="opacity:.55;font-size:11px">Observe / Decide / Action</span>';
    this.root.appendChild(header);

    this.list = document.createElement("div");
    this.list.style.cssText = "flex:1 1 auto;overflow-y:auto;padding:8px 12px;line-height:1.5";
    this.root.appendChild(this.list);

    // Override lives in the panel's own footer so it cannot be absent (FR-16).
    const footer = document.createElement("div");
    footer.style.cssText =
      "flex:0 0 auto;padding:10px 12px;border-top:1px solid #2a2e3f;display:flex;align-items:center;gap:10px";
    this.overrideButton = document.createElement("button");
    this.overrideButton.style.cssText =
      "background:#f7768e;color:#1a1b26;border:0;border-radius:4px;padding:6px 14px;font:600 12px ui-monospace,monospace;cursor:pointer";
    this.statusLabel = document.createElement("span");
    this.statusLabel.style.cssText = "opacity:.7";
    footer.append(this.overrideButton, this.statusLabel);
    this.root.appendChild(footer);

    this.overrideButton.addEventListener("click", () => this.setPaused(!this.paused));
    this.renderOverride();
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    this.renderOverride();
    this.onOverrideChange?.(paused);
  }

  get isPaused(): boolean {
    return this.paused;
  }

  private renderOverride(): void {
    this.overrideButton.textContent = this.paused ? "RESUME" : "PAUSE";
    this.overrideButton.style.background = this.paused ? "#9ece6a" : "#f7768e";
    this.statusLabel.textContent = this.paused
      ? "supervisor override — output held"
      : "running — override always available";
  }

  /** Replace all rows. */
  setEntries(entries: LogEntry[]): void {
    this.entries = [];
    this.list.innerHTML = "";
    for (const entry of entries) this.append(entry);
  }

  /** Append one row and keep the newest visible. */
  append(entry: LogEntry): void {
    this.entries.push(entry);
    const time = new Date(entry.timestamp * 1000).toLocaleTimeString("en-GB", { hour12: false });

    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:8px;margin-bottom:2px";
    row.innerHTML =
      `<span style="opacity:.45;flex:0 0 auto">${time}</span>` +
      `<span style="color:${STAGE_COLOUR[entry.stage]};flex:0 0 58px;font-weight:600">${entry.stage}</span>` +
      `<span style="flex:1 1 auto">${emphasise(entry.detail)}</span>`;
    this.list.appendChild(row);
    this.list.scrollTop = this.list.scrollHeight;
  }

  get count(): number {
    return this.entries.length;
  }
}

export async function loadLog(url: string): Promise<LogPayload> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`decision log ${url}: HTTP ${response.status}`);
  return (await response.json()) as LogPayload;
}
