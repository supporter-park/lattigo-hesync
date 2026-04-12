package hesync

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

const (
	galoisKeyPrefix = "gk_"
	galoisKeySuffix = ".bin"
	relinKeyFile    = "rlk.bin"
)

// SerializeEVKSet writes each evaluation key in the set to individual files
// in the specified directory. Each GaloisKey is written to gk_<galEl>.bin
// and the RelinearizationKey (if present) to rlk.bin.
//
// This enables HESync to load individual keys on-demand from disk rather
// than keeping all keys in memory simultaneously.
func SerializeEVKSet(evkSet *rlwe.MemEvaluationKeySet, basePath string) error {

	if err := os.MkdirAll(basePath, 0755); err != nil {
		return fmt.Errorf("SerializeEVKSet: mkdir %s: %w", basePath, err)
	}

	// Serialize RelinearizationKey
	if evkSet.RelinearizationKey != nil {
		path := filepath.Join(basePath, relinKeyFile)
		if err := writeKeyToFile(path, evkSet.RelinearizationKey); err != nil {
			return fmt.Errorf("SerializeEVKSet: write relin key: %w", err)
		}
	}

	// Serialize each GaloisKey individually
	for galEl, gk := range evkSet.GaloisKeys {
		filename := fmt.Sprintf("%s%d%s", galoisKeyPrefix, galEl, galoisKeySuffix)
		path := filepath.Join(basePath, filename)
		if err := writeKeyToFile(path, gk); err != nil {
			return fmt.Errorf("SerializeEVKSet: write galois key %d: %w", galEl, err)
		}
	}

	return nil
}

// writeKeyToFile writes an object implementing io.WriterTo to a file.
func writeKeyToFile(path string, key interface{ BinarySize() int; MarshalBinary() ([]byte, error) }) error {
	data, err := key.MarshalBinary()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// EVKIndex maps key identifiers to file paths on disk.
type EVKIndex struct {
	BasePath     string
	GaloisKeys   map[uint64]string // galEl -> file path
	RelinKeyPath string            // empty if no relin key
}

// BuildIndex scans a directory for serialized EVK files and builds an index
// mapping Galois elements to their file paths.
func BuildIndex(basePath string) (*EVKIndex, error) {
	idx := &EVKIndex{
		BasePath:   basePath,
		GaloisKeys: make(map[uint64]string),
	}

	// Check for relinearization key
	rlkPath := filepath.Join(basePath, relinKeyFile)
	if _, err := os.Stat(rlkPath); err == nil {
		idx.RelinKeyPath = rlkPath
	}

	// Scan for galois key files
	entries, err := os.ReadDir(basePath)
	if err != nil {
		return nil, fmt.Errorf("BuildIndex: read dir %s: %w", basePath, err)
	}

	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, galoisKeyPrefix) && strings.HasSuffix(name, galoisKeySuffix) {
			// Extract galois element from filename
			numStr := strings.TrimPrefix(name, galoisKeyPrefix)
			numStr = strings.TrimSuffix(numStr, galoisKeySuffix)

			galEl, err := strconv.ParseUint(numStr, 10, 64)
			if err != nil {
				continue // skip files with non-numeric names
			}

			idx.GaloisKeys[galEl] = filepath.Join(basePath, name)
		}
	}

	return idx, nil
}

// LoadGaloisKey loads a single GaloisKey from disk.
func LoadGaloisKey(path string) (*rlwe.GaloisKey, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadGaloisKey: open %s: %w", path, err)
	}
	defer f.Close()

	gk := new(rlwe.GaloisKey)
	if _, err := gk.ReadFrom(f); err != nil {
		return nil, fmt.Errorf("LoadGaloisKey: read %s: %w", path, err)
	}

	return gk, nil
}

// LoadRelinearizationKey loads a RelinearizationKey from disk.
func LoadRelinearizationKey(path string) (*rlwe.RelinearizationKey, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadRelinearizationKey: open %s: %w", path, err)
	}
	defer f.Close()

	rlk := new(rlwe.RelinearizationKey)
	if _, err := rlk.ReadFrom(f); err != nil {
		return nil, fmt.Errorf("LoadRelinearizationKey: read %s: %w", path, err)
	}

	return rlk, nil
}

// GetGaloisElementsList returns all Galois elements available in the index.
func (idx *EVKIndex) GetGaloisElementsList() []uint64 {
	galEls := make([]uint64, 0, len(idx.GaloisKeys))
	for galEl := range idx.GaloisKeys {
		galEls = append(galEls, galEl)
	}
	return galEls
}
