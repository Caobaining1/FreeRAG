package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The chart service as a child process.
//
// The kernel owns it rather than asking the user to start one, because the thing
// it needs is not a server but 7.4 GB of resident weights: a service nobody
// started is a backend that silently does nothing, and "figures are not
// described" is indistinguishable from "there were no figures".
//
// Owning it also means owning its lifetime. A chart service left running keeps
// those weights after the app closes, which shows up as memory pressure in
// whatever the user does next — so it is killed on the way out, not left to be
// reaped.

const (
	// chartServiceStart bounds the wait for the service to answer /health.
	//
	// It is generous because starting up means importing torch, which is slow
	// and slower on a cold page cache — but it is NOT waiting for the weights:
	// the service loads them on the first transcription, so this stays bounded
	// even though a load takes a minute.
	chartServiceStart = 2 * time.Minute
	// chartServiceStop is how long the process is given to go away before it
	// is killed. It has nothing to persist, so this is only politeness.
	chartServiceStop = 5 * time.Second
)

// chartService is a running chartvlm/server.py owned by this process.
type chartService struct {
	endpoint string
	cmd      *exec.Cmd
	port     int
}

// startChartService launches the service on a free port and waits for it.
//
// The port is picked here rather than fixed, so two kernels — a dev build and
// the desktop app, say — cannot collide on one and then have the loser silently
// talk to the winner's model.
// minChartMemoryBytes is what a machine needs to hold the chart model without
// swapping. The weights are 7.4 GB resident in fp32 and 3.7 GB in fp16, over
// and above Qdrant, the parse sidecar and whatever else is running.
//
// 12 GB sits between the two on purpose: it admits fp16 comfortably and refuses
// an 8 GB machine outright, which is the case that matters — there the load does
// not fail, it swaps, and the desktop stops responding while indexing runs.
const minChartMemoryBytes = 12 << 30

// checkChartMemory refuses to start the chart service on a machine that cannot
// hold it.
//
// The total is the shell's, measured by the desktop's hardware scan at startup,
// rather than the kernel's own: "how much memory does this machine have" has no
// portable answer in the standard library, and the shell has already asked.
//
// A machine that reported nothing is allowed through — the kernel also runs
// headless, in a terminal and in tests, where no shell scanned anything and
// refusing would be a guess dressed as a measurement.
func checkChartMemory() error {
	if os.Getenv("FREERAG_CHART_FORCE") != "" {
		return nil
	}
	raw := os.Getenv("FREERAG_HW_MEMORY_BYTES")
	if raw == "" {
		return nil
	}
	total, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || total == 0 {
		return nil
	}
	if total < minChartMemoryBytes {
		return fmt.Errorf("this machine reports %.1f GB of memory and the chart model wants about %d GB to load without swapping; FREERAG_CHART_FORCE=1 overrides this",
			float64(total)/(1<<30), minChartMemoryBytes>>30)
	}
	return nil
}

func startChartService() (*chartService, error) {
	python, script, err := discoverChartService()
	if err != nil {
		return nil, err
	}

	port, err := freeChartPort()
	if err != nil {
		return nil, fmt.Errorf("no free port for the chart service: %w", err)
	}

	args := []string{script, "--port", strconv.Itoa(port)}
	// Passed explicitly rather than through the environment so the service
	// cannot start up pointing somewhere else than the kernel thinks it does.
	if root := strings.TrimSpace(os.Getenv("FREERAG_CHARTVLM_ROOT")); root != "" {
		args = append(args, "--root", root)
	}

	cmd := exec.Command(python, args...)
	cmd.Env = os.Environ() // the backbone overrides travel as FREERAG_CHARTVLM_*
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("chart service stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("could not start the chart service (%s %s): %w", python, script, err)
	}

	service := &chartService{cmd: cmd, port: port, endpoint: fmt.Sprintf("http://127.0.0.1:%d", port)}
	go service.logStderr(stderr)

	// The wait is what turns a hundred possible failures — a missing venv, an
	// import error, a bad root — into one message here instead of one per
	// figure later.
	if !service.waitReady(chartServiceStart) {
		_ = service.Close()
		return nil, fmt.Errorf("the chart service did not answer /health within %s (see its log above)",
			chartServiceStart)
	}
	return service, nil
}

// Close stops the service and releases its weights.
func (s *chartService) Close() error {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return nil
	}
	if err := s.cmd.Process.Kill(); err != nil {
		return err
	}
	// Reaped rather than left: an unreaped child holds its exit status and
	// keeps a slot in the process table until this process ends anyway.
	_, _ = s.cmd.Process.Wait()
	s.cmd = nil
	return nil
}

// waitReady polls /health until the service answers or the budget runs out.
func (s *chartService) waitReady(budget time.Duration) bool {
	probe := &chartClient{endpoint: s.endpoint, timeout: 5 * time.Second}
	deadline := time.Now().Add(budget)
	for {
		if probe.Reachable(context.Background()) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		// Straight lines, not backoff: the wait is bounded and short-lived, and
		// the interesting case is a service that never comes up at all, which
		// backoff would only report later.
		time.Sleep(250 * time.Millisecond)
	}
}

// logStderr forwards the service's stderr into the kernel's log.
//
// Its failures are the ones a user can act on — no torch, weights missing, a
// transformers version that silently mismatches the checkpoint — and they are
// all reported on this stream. Swallowing it would leave "figures are not
// described" as the only symptom.
func (s *chartService) logStderr(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 8*1024), 1<<20)
	for scanner.Scan() {
		if line := strings.TrimSpace(scanner.Text()); line != "" {
			log.Printf("chartvlm: %s", line)
		}
	}
}

// ownedSuffix says in the startup line whether the service is ours.
//
// Spelled out because it decides who has to clean up: a service this session
// started is stopped with it, and one that was already listening is not — so
// the log has to make clear which of the two is in play.
func ownedSuffix(service *chartService) string {
	if service == nil {
		return " (an already-running service; not stopped with this process)"
	}
	return fmt.Sprintf(" (started by this process, port %d)", service.port)
}

// discoverChartService resolves the interpreter and script, the same way the
// parse sidecar does: explicit overrides first, then the repository, then
// whatever is on PATH.
func discoverChartService() (python, script string, err error) {
	if script = strings.TrimSpace(os.Getenv("FREERAG_CHART_SERVER")); script != "" {
		if python = strings.TrimSpace(os.Getenv("FREERAG_CHART_PYTHON")); python == "" {
			python = "python3"
		}
		return python, script, nil
	}

	// Refused here rather than at the first figure: without the checkout there
	// is no model, and a service that starts and then fails every transcription
	// would turn one clear misconfiguration into one error per figure.
	if strings.TrimSpace(os.Getenv("FREERAG_CHARTVLM_ROOT")) == "" {
		return "", "", errors.New("FREERAG_CHARTVLM_ROOT is not set: point it at the Laya-Chart checkout")
	}

	root, err := chartRepoRoot()
	if err != nil {
		return "", "", err
	}
	script = filepath.Join(root, "chartvlm", "server.py")
	if _, err := os.Stat(script); err != nil {
		return "", "", fmt.Errorf("chart service not found at %s", script)
	}

	// The chart venv first: torch is only installed there, and falling back to
	// a python3 without it produces a service that starts and then cannot load
	// anything.
	python = strings.TrimSpace(os.Getenv("FREERAG_CHART_PYTHON"))
	if python == "" {
		for _, candidate := range []string{
			filepath.Join(root, ".venv-chartvlm", "bin", "python"),
			filepath.Join(root, ".venv314", "bin", "python"),
		} {
			if _, err := os.Stat(candidate); err == nil {
				python = candidate
				break
			}
		}
	}
	if python == "" {
		python = "python3"
	}
	return python, script, nil
}

// freeChartPort reserves and immediately releases a port.
//
// The gap between releasing it and the service binding it is a race, and it is
// accepted: the alternative is letting the service pick and then parsing its
// output for the port, which is slower, fragile, and still not race-free.
func freeChartPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := listener.Addr().(*net.TCPAddr).Port
	closeErr := listener.Close()
	if closeErr != nil {
		return 0, closeErr
	}
	return port, nil
}

// chartRepoRoot walks up from the executable and the working directory looking
// for the repository.
//
// It looks for the parse sidecar rather than for chartvlm/server.py because the
// sidecar is what every layout has — dev checkout and packaged app alike —
// whereas chartvlm/ is only present in a source tree.
func chartRepoRoot() (string, error) {
	starts := make([]string, 0, 2)
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	for _, start := range starts {
		dir := start
		for range 5 {
			if _, err := os.Stat(filepath.Join(dir, "sidecar", "parse_server.py")); err == nil {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", errors.New("could not locate the freerag repository root (set FREERAG_CHART_SERVER)")
}
