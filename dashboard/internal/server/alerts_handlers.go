package server

import (
	"net/http"
	"strconv"

	"github.com/starfederation/datastar-go/datastar"
)

// handleAckAlert acknowledges one alert (?id=N) or all (no id). This edits
// dashboard metadata only; it never signals or steers a trainer.
func (s *Server) handleAckAlert(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
	sse := datastar.NewSSE(w, r)
	if err := s.db.AckAlert(id); err != nil {
		toastErr(sse, "ack failed: "+err.Error())
		return
	}
	if active, err := s.db.ActiveAlerts(20); err == nil {
		_ = sse.PatchElements(renderAlerts(active))
	}
}
