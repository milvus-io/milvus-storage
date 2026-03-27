.PHONY: fix-checks fix-format fix-tidy

fix-checks:
	$(MAKE) fix-tidy
	$(MAKE) fix-format

fix-format:
	$(MAKE) -C cpp fix-format
	$(MAKE) -C python fix-format

fix-tidy:
	$(MAKE) -C cpp fix-tidy
