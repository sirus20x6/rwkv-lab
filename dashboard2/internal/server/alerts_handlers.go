package server

import (
	"net/http"
	"strconv"

	"github.com/starfederation/datastar-go/datastar"
)

// handleAckAlert acknowledges one alert (?id=N) or all (no id).
func (s *Server) handleAckAlert(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
	sse := datastar.NewSSE(w, r)
	if err := s.db.AckAlert(id); err != nil {
		toast(sse, "ack failed: "+err.Error())
		return
	}
	// Repaint the banner immediately (don't wait for the next tick).
	if active, err := s.db.ActiveAlerts(20); err == nil {
		_ = sse.PatchElements(renderAlerts(active, s.autoStopOn()))
	}
}

// handleAutoStop toggles the opt-in auto-stop behavior.
func (s *Server) handleAutoStop(w http.ResponseWriter, r *http.Request) {
	on := r.URL.Query().Get("on") == "1"
	sse := datastar.NewSSE(w, r)
	if s.detector != nil {
		s.detector.SetAutoStop(on)
	}
	s.db.LogAction(nowTs(), "autostop", "", "", boolStr(on), 0)
	if active, err := s.db.ActiveAlerts(20); err == nil {
		_ = sse.PatchElements(renderAlerts(active, on))
	}
	if on {
		toast(sse, "auto-stop ENABLED — critical alerts will SIGINT the run")
	} else {
		toast(sse, "auto-stop disabled")
	}
}

func (s *Server) autoStopOn() bool {
	return s.detector != nil && s.detector.AutoStop()
}

func boolStr(b bool) string {
	if b {
		return "on"
	}
	return "off"
}
