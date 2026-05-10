name: Pull Request
title: ""
labels: []
body:

  - type: markdown
    attributes:
      value: |
        ## Description
        <!-- Describe your changes -->

  - type: textarea
    id: type-of-change
    attributes:
      label: "Type of Change"
      description: "What type of change does your code introduce?"
      placeholder: |
        - [ ] Bug fix
        - [ ] New feature
        - [ ] Code refactoring
        - [ ] Documentation update
        - [ ] Other
    validations:
      required: true

  - type: textarea
    id: test-plan
    attributes:
      label: "Test Plan"
      description: "Describe the tests you ran to verify your changes"
      placeholder: |
        - [ ] Unit tests added
        - [ ] Manual testing performed
