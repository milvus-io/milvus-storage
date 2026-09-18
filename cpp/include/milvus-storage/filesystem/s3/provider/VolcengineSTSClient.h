#pragma once

#include <aws/core/internal/AWSHttpResourceClient.h>

namespace milvus_storage {

class AWS_CORE_API VolcengineSTSCredentialsClient : public ::Aws::Internal::AWSHttpResourceClient {
  public:
  explicit VolcengineSTSCredentialsClient(const Aws::Client::ClientConfiguration& clientConfiguration);

  VolcengineSTSCredentialsClient& operator=(VolcengineSTSCredentialsClient& rhs) = delete;
  VolcengineSTSCredentialsClient(const VolcengineSTSCredentialsClient& rhs) = delete;
  VolcengineSTSCredentialsClient& operator=(VolcengineSTSCredentialsClient&& rhs) = delete;
  VolcengineSTSCredentialsClient(const VolcengineSTSCredentialsClient&& rhs) = delete;

  struct STSAssumeRoleWithWebIdentityRequest {
      Aws::String webIdentityToken;
      Aws::String roleArn;
      Aws::String roleSessionName;
  };

  struct STSAssumeRoleWithWebIdentityResult {
    Aws::Auth::AWSCredentials creds;
  };

  STSAssumeRoleWithWebIdentityResult GetAssumeRoleWithWebIdentityCredentials(
      const STSAssumeRoleWithWebIdentityRequest& request);

  // Step-2 of the cross-account chain: a plain sts:AssumeRole. The caller is
  // itself an STS-temporary identity (the step-1 AssumeRoleWithOIDC result), so
  // callerSecurityToken is mandatory. Unlike AssumeRoleWithOIDC (where the OIDC
  // token *is* the credential and the request is unsigned), this request is
  // signed with Volcengine V4 (HMAC-SHA256) using the caller's AK/SK.
  struct STSAssumeRoleRequest {
      Aws::String callerAccessKeyId;
      Aws::String callerAccessKeySecret;
      Aws::String callerSecurityToken;  // step-1 SessionToken; goes in X-Security-Token
      Aws::String roleTrn;              // customer target role (extfs.role_arn)
      Aws::String roleSessionName;
  };

  struct STSAssumeRoleResult {
    Aws::Auth::AWSCredentials creds;
  };

  // Returns empty credentials on failure; errors are logged.
  STSAssumeRoleResult GetAssumeRoleCredentials(const STSAssumeRoleRequest& request);

  // --- Volcengine V4 signing (exposed for golden-vector testing) -----------
  // Result of signing a request with Volcengine V4 (HMAC-SHA256).
  struct V4SignResult {
    Aws::String authorization;     // full Authorization header value
    Aws::String signature;         // lowercase hex signature only
    Aws::String signedHeaders;     // e.g. "content-type;host;x-content-sha256;x-date"
    Aws::String xContentSha256;    // hex SHA256 of the body (X-Content-Sha256 header)
  };

  // Signs a POST to path "/" that carries `canonicalQuery` in the URL and
  // `body` as an x-www-form-urlencoded payload, using Volcengine V4. The
  // timestamps (xDate = YYYYMMDD'T'HHMMSS'Z', scopeDate = YYYYMMDD) are
  // injected rather than read from the clock, so the output is deterministic
  // and unit-testable; production passes the current UTC time. Static because
  // signing needs no client state. This is the single source of truth for the
  // 5 Volcengine-vs-AWS-SigV4 differences (see the .cpp for the enumeration).
  static V4SignResult SignRequestV4(const Aws::String& accessKeyId,
                                    const Aws::String& secretAccessKey,
                                    const Aws::String& region,
                                    const Aws::String& service,
                                    const Aws::String& host,
                                    const Aws::String& canonicalQuery,
                                    const Aws::String& body,
                                    const Aws::String& contentType,
                                    const Aws::String& xDate,
                                    const Aws::String& scopeDate);

  private:
  Aws::String m_endpoint;
  Aws::String m_region;   // Volcengine V4 credential-scope region; default cn-beijing
  Aws::String m_service;  // "sts"
};
}  // namespace milvus_storage
