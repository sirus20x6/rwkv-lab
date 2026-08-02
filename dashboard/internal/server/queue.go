package server

import (
	"fmt"
	"strings"

	"trainboard/internal/db"
)

// renderQueue preserves the historical queue as a read-only projection. Queue
// lifecycle belongs to TrainVM; the dashboard no longer enqueues, starts,
// cancels, reconciles, or auto-starts generic trainer processes.
func renderQueue(items []db.QueueItem, _ bool, _ bool) string {
	var b strings.Builder
	b.WriteString(`<div id="queue-list">`)
	b.WriteString(`<div class="queue-bar"><span class="muted">legacy queue history · read only</span></div>`)
	if len(items) == 0 {
		b.WriteString(`<div class="empty">queue empty</div>`)
	}
	for _, q := range items {
		pid := ""
		if q.PID != nil {
			pid = fmt.Sprintf(" · PID %d", *q.PID)
		}
		fmt.Fprintf(&b,
			`<div class="queue-item %s"><span class="q-status">%s</span>`+
				`<span class="q-cmd">%s %s%s</span></div>`,
			esc(q.Status), esc(q.Status), esc(q.Script), esc(q.Args), pid)
	}
	b.WriteString(`</div>`)
	return b.String()
}

// renderLaunchHistory fills the datalist backing the launch-args input with
// recently launched/queued arg strings (from the actions audit table).
func renderLaunchHistory(items []db.LaunchHistoryItem) string {
	var b strings.Builder
	b.WriteString(`<datalist id="launch-history">`)
	for _, it := range items {
		fmt.Fprintf(&b, `<option value="%s" label="%s"></option>`, esc(it.Args), esc(it.Script))
	}
	b.WriteString(`</datalist>`)
	return b.String()
}
