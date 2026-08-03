package server

// Legacy live tuning wrote directly into the dashboard database for trainers
// to poll. TrainVM now owns typed, revisioned controls through hostd, so this
// unregistered compatibility stub is deliberately incapable of mutation.

import "net/http"

func (s *Server) handleSetControl(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "legacy live tuning is retired; use typed TrainVM controls", http.StatusGone)
}
