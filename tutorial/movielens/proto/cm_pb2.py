# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cm.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import item_info_pb2 as item__info__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='cm.proto',
  package='cm',
  syntax='proto3',
  serialized_options=b'\200\001\000',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x08\x63m.proto\x12\x02\x63m\x1a\x0fitem_info.proto\"-\n\tCMRequest\x12\x0e\n\x06log_id\x18\x01 \x01(\t\x12\x10\n\x08item_ids\x18\x02 \x03(\t\"\x7f\n\nCMResponse\x12#\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x14.cm.CMResponse.Error\x12\'\n\nitem_infos\x18\x02 \x03(\x0b\x32\x13.item_info.ItemInfo\x1a#\n\x05\x45rror\x12\x0c\n\x04\x63ode\x18\x01 \x01(\r\x12\x0c\n\x04text\x18\x02 \x01(\t25\n\tCMService\x12(\n\x07\x63m_call\x12\r.cm.CMRequest\x1a\x0e.cm.CMResponseB\x03\x80\x01\x00\x62\x06proto3'
  ,
  dependencies=[item__info__pb2.DESCRIPTOR,])




_CMREQUEST = _descriptor.Descriptor(
  name='CMRequest',
  full_name='cm.CMRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='log_id', full_name='cm.CMRequest.log_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='item_ids', full_name='cm.CMRequest.item_ids', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=78,
)


_CMRESPONSE_ERROR = _descriptor.Descriptor(
  name='Error',
  full_name='cm.CMResponse.Error',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='cm.CMResponse.Error.code', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='text', full_name='cm.CMResponse.Error.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=172,
  serialized_end=207,
)

_CMRESPONSE = _descriptor.Descriptor(
  name='CMResponse',
  full_name='cm.CMResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='error', full_name='cm.CMResponse.error', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='item_infos', full_name='cm.CMResponse.item_infos', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_CMRESPONSE_ERROR, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=80,
  serialized_end=207,
)

_CMRESPONSE_ERROR.containing_type = _CMRESPONSE
_CMRESPONSE.fields_by_name['error'].message_type = _CMRESPONSE_ERROR
_CMRESPONSE.fields_by_name['item_infos'].message_type = item__info__pb2._ITEMINFO
DESCRIPTOR.message_types_by_name['CMRequest'] = _CMREQUEST
DESCRIPTOR.message_types_by_name['CMResponse'] = _CMRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CMRequest = _reflection.GeneratedProtocolMessageType('CMRequest', (_message.Message,), {
  'DESCRIPTOR' : _CMREQUEST,
  '__module__' : 'cm_pb2'
  # @@protoc_insertion_point(class_scope:cm.CMRequest)
  })
_sym_db.RegisterMessage(CMRequest)

CMResponse = _reflection.GeneratedProtocolMessageType('CMResponse', (_message.Message,), {

  'Error' : _reflection.GeneratedProtocolMessageType('Error', (_message.Message,), {
    'DESCRIPTOR' : _CMRESPONSE_ERROR,
    '__module__' : 'cm_pb2'
    # @@protoc_insertion_point(class_scope:cm.CMResponse.Error)
    })
  ,
  'DESCRIPTOR' : _CMRESPONSE,
  '__module__' : 'cm_pb2'
  # @@protoc_insertion_point(class_scope:cm.CMResponse)
  })
_sym_db.RegisterMessage(CMResponse)
_sym_db.RegisterMessage(CMResponse.Error)


DESCRIPTOR._options = None

_CMSERVICE = _descriptor.ServiceDescriptor(
  name='CMService',
  full_name='cm.CMService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=209,
  serialized_end=262,
  methods=[
  _descriptor.MethodDescriptor(
    name='cm_call',
    full_name='cm.CMService.cm_call',
    index=0,
    containing_service=None,
    input_type=_CMREQUEST,
    output_type=_CMRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_CMSERVICE)

DESCRIPTOR.services_by_name['CMService'] = _CMSERVICE

# @@protoc_insertion_point(module_scope)
