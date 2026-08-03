package server

import (
	"strings"
	"testing"

	"trainboard/internal/db"
)

func TestLegacyQueueProjectionIsReadOnly(t *testing.T) {
	pid := int64(42)
	html := renderQueue([]db.QueueItem{{
		ID: 7, Script: "legacy.py", Args: "--steps 10", Status: "queued", PID: &pid,
	}}, true, true)
	if !strings.Contains(html, "legacy queue history · read only") ||
		!strings.Contains(html, "legacy.py") || !strings.Contains(html, "--steps 10") {
		t.Fatalf("read-only queue history is incomplete: %s", html)
	}
	for _, forbidden := range []string{"/api/queue/", "start next", "data-on:click"} {
		if strings.Contains(html, forbidden) {
			t.Fatalf("read-only queue exposes %q: %s", forbidden, html)
		}
	}
}

func TestAlertProjectionHasNoTrainerAuthority(t *testing.T) {
	html := renderAlerts([]db.Alert{{
		ID: 9, RunName: "vision", Kind: "stall", Severity: "warn", Message: "no progress",
	}})
	if !strings.Contains(html, "health findings · read only") ||
		!strings.Contains(html, "/api/alerts/ack?id=9") {
		t.Fatalf("read-only alert metadata controls are incomplete: %s", html)
	}
	for _, forbidden := range []string{"/api/autostop", "auto-stop", "SIGINT"} {
		if strings.Contains(html, forbidden) {
			t.Fatalf("alert projection exposes retired authority %q: %s", forbidden, html)
		}
	}
}
