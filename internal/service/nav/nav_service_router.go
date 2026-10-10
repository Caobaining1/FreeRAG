// The production nav-tree router, the FreeRAG mirror of RAGFlow's
// NavServiceRouter. It routes a query to the documents that best match it via
// the installed NavService (store-backed in FreeRAG).
package nav

import (
	"context"
	"strings"

	"go.uber.org/zap"

	"freerag/internal/common"
)

const navServiceMaxDocs = 12

// NavServiceRouter routes through the installed navigation-tree service.
type NavServiceRouter struct{}

// Route implements the agentic NavTreeRouter contract. It returns
// (nil, nil) when no NavService is installed — treated as "no compiled tree".
func (NavServiceRouter) Route(ctx context.Context, tenantID, kbID, query string, docScope []string, topK int) ([][2]string, error) {
	svc := GetNavService()
	if svc == nil {
		return nil, nil // no compiled tree available
	}
	if strings.TrimSpace(query) == "" || kbID == "" {
		return nil, nil
	}
	if topK <= 0 {
		topK = navServiceMaxDocs
	}

	hits, err := svc.Search(ctx, tenantID, kbID, query, nil, docScope, topK)
	if err != nil {
		common.Warn("dataset navigation search: NavService.Search failed",
			zap.String("kb_id", kbID), zap.Error(err))
		return nil, err
	}
	routed := make([][2]string, 0, len(hits))
	for _, h := range hits {
		if h.Type == TypeNavCluster {
			continue
		}
		did := strings.TrimSpace(h.DocID)
		if did == "" {
			continue
		}
		routed = append(routed, [2]string{did, strings.TrimSpace(h.Name)})
	}
	return routed, nil
}

func mapToFields(kbID string, err error) []interface{} {
	return []interface{}{"kb_id", kbID, "error", err}
}

// NewNavServiceRouter returns the production router.
func NewNavServiceRouter() *NavServiceRouter { return &NavServiceRouter{} }
