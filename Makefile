.PHONY: fix-checks fix-format fix-tidy

fix-checks: fix-format fix-tidy

fix-format:
	$(MAKE) -C cpp fix-format
	$(MAKE) -C python fix-format

fix-tidy:
	$(MAKE) -C cpp fix-tidy
