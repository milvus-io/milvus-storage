package fs

import (
	"context"
	"fmt"
	"net/url"

	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"go.uber.org/zap"
)

type MinioFs struct {
	client     *minio.Client
	bucketName string
}

func (fs *MinioFs) OpenFile(path string) (file.File, error) {
	return file.NewMinioFile(fs.client, path, fs.bucketName)
}

func (fs *MinioFs) Rename(src string, dst string) error {
	_, err := fs.client.CopyObject(context.TODO(), minio.CopyDestOptions{Bucket: fs.bucketName, Object: dst}, minio.CopySrcOptions{Bucket: fs.bucketName, Object: src})
	if err != nil {
		return err
	}
	err = fs.client.RemoveObject(context.TODO(), fs.bucketName, src, minio.RemoveObjectOptions{})
	if err != nil {
		log.Warn("failed to remove source object", log.String("source", src))
	}
	return nil
}

func (fs *MinioFs) DeleteFile(path string) error {
	return fs.client.RemoveObject(context.TODO(), fs.bucketName, path, minio.RemoveObjectOptions{})
}

func (fs *MinioFs) CreateDir(path string) error {
	return nil
}

func (fs *MinioFs) List(path string) ([]FileEntry, error) {
	ret := make([]FileEntry, 0)
	for objInfo := range fs.client.ListObjects(context.TODO(), fs.bucketName, minio.ListObjectsOptions{Prefix: path}) {
		if objInfo.Err != nil {
			log.Warn("list object error", zap.Error(objInfo.Err))
			return nil, objInfo.Err
		}
		ret = append(ret, FileEntry{Path: objInfo.Key})
	}
	return ret, nil
}

func (fs *MinioFs) ReadFile(path string) ([]byte, error) {
	obj, err := fs.client.GetObject(context.TODO(), fs.bucketName, path, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}

	stat, err := obj.Stat()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, 0, stat.Size)
	n, err := obj.Read(buf)
	if err != nil {
		return nil, err
	}
	if n != int(stat.Size) {
		return nil, fmt.Errorf("failed to read full file, expect: %d, actual: %d", stat.Size, n)
	}
	return buf, nil
}

// uri should be s3://accessKey:secretAceessKey@endpoint/bucket/
func NewMinioFs(uri *url.URL) (*MinioFs, error) {
	accessKey := uri.User.Username()
	secretAccessKey, set := uri.User.Password()
	if !set {
		log.Warn("secret access key not set")
	}
	cli, err := minio.New(uri.Host, &minio.Options{
		BucketLookup: minio.BucketLookupAuto,
		Creds:        credentials.NewStaticV4(accessKey, secretAccessKey, ""),
	})
	if err != nil {
		return nil, err
	}

	exist, err := cli.BucketExists(context.TODO(), uri.Path)
	if err != nil {
		return nil, err
	}

	if !exist {
		if err = cli.MakeBucket(context.TODO(), uri.Path, minio.MakeBucketOptions{}); err != nil {
			return nil, err
		}
	}

	return &MinioFs{
		client:     cli,
		bucketName: uri.Path,
	}, nil
}
