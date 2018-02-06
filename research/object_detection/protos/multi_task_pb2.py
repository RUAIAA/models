# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/multi_task.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/multi_task.proto',
  package='',
  serialized_pb=_b('\n(object_detection/protos/multi_task.proto\"h\n\x0eMultiTaskClass\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0b\n\x03num\x18\x02 \x02(\x05\x12#\n\x1b\x61ssociated_with_box_codings\x18\x03 \x02(\x08\x12\x16\n\x0ehas_background\x18\x04 \x01(\x08')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_MULTITASKCLASS = _descriptor.Descriptor(
  name='MultiTaskClass',
  full_name='MultiTaskClass',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='MultiTaskClass.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num', full_name='MultiTaskClass.num', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='associated_with_box_codings', full_name='MultiTaskClass.associated_with_box_codings', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_background', full_name='MultiTaskClass.has_background', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=148,
)

DESCRIPTOR.message_types_by_name['MultiTaskClass'] = _MULTITASKCLASS

MultiTaskClass = _reflection.GeneratedProtocolMessageType('MultiTaskClass', (_message.Message,), dict(
  DESCRIPTOR = _MULTITASKCLASS,
  __module__ = 'object_detection.protos.multi_task_pb2'
  # @@protoc_insertion_point(class_scope:MultiTaskClass)
  ))
_sym_db.RegisterMessage(MultiTaskClass)


# @@protoc_insertion_point(module_scope)
