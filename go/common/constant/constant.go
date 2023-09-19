package constant

const (
	ReadBatchSize          = 1024
	ManifestTempFileSuffix = ".manifest.tmp"
	ManifestFileSuffix     = ".manifest"
	ManifestDir            = "versions"
	BlobDir                = "blobs"
	ParquetDataFileSuffix  = ".parquet"
	OffsetFieldName        = "__offset"
	VectorDataDir          = "vector"
	ScalarDataDir          = "scalar"
	DeleteDataDir          = "delete"
	LatestManifestVersion  = -1
)
