# Define common headers directory for examples
common_inc_dir = include_directories('common')

# Compile examples
app_examples = [
  'list_parameters',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach

if get_option('enable-tensorflow')
  subdir('tensorflow')
endif

if get_option('enable-tflite')
  subdir('tflite')
endif

if get_option('enable-tensorrt')
  subdir('tensorrt')
endif
