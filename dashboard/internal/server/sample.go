package server

// Legacy checkpoint sampling used to rebuild a model and invoke Python inside
// the dashboard request. Generation is not a read-only inspection operation and
// no descriptor-backed replacement exists yet, so the route remains fail-closed
// until it is removed from the central router.

import "net/http"

func (s *Server) handleSample(w http.ResponseWriter, _ *http.Request) {
	http.Error(w, "dashboard checkpoint generation is retired; no descriptor-backed operation is available", http.StatusGone)
}
