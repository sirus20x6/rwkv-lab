package sysmon

import (
	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/disk"
	"github.com/shirou/gopsutil/v4/load"
	"github.com/shirou/gopsutil/v4/mem"
)

const gb = 1024 * 1024 * 1024

// primeCPU establishes the baseline for non-blocking cpu.Percent(0,…) calls.
func primeCPU() {
	_, _ = cpu.Percent(0, false)
	_, _ = cpu.Percent(0, true)
}

// readHost samples CPU/RAM/disk/load. diskPath is the mount to report free space
// for (the moe-mla repo lives on /thearray).
func readHost(diskPath string) Host {
	h := Host{}

	if pcts, err := cpu.Percent(0, false); err == nil && len(pcts) > 0 {
		h.CPUPct = pcts[0]
	}
	if per, err := cpu.Percent(0, true); err == nil {
		h.CPUPerCore = per
		h.CPUCount = len(per)
	}
	if vm, err := mem.VirtualMemory(); err == nil {
		h.RAMUsedGB = float64(vm.Used) / gb
		h.RAMTotalGB = float64(vm.Total) / gb
		h.RAMPct = vm.UsedPercent
	}
	if du, err := disk.Usage(diskPath); err == nil {
		h.DiskUsedGB = float64(du.Used) / gb
		h.DiskFreeGB = float64(du.Free) / gb
		h.DiskTotalGB = float64(du.Total) / gb
		h.DiskPct = du.UsedPercent
	}
	if l, err := load.Avg(); err == nil {
		h.Load1, h.Load5, h.Load15 = l.Load1, l.Load5, l.Load15
	}
	return h
}
