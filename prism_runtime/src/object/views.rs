//! Python-visible view objects over internal runtime metadata.

use crate::object::class::PyClassObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::Cell;
use prism_code::CodeObject;
use prism_core::Value;
use prism_core::intern::InternedString;
use std::sync::Arc;

#[repr(C)]
pub struct CodeObjectView {
    header: ObjectHeader,
    code: Arc<CodeObject>,
}

impl CodeObjectView {
    #[inline]
    pub fn new(code: Arc<CodeObject>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::CODE),
            code,
        }
    }

    #[inline]
    pub fn code(&self) -> &Arc<CodeObject> {
        &self.code
    }
}

impl PyObject for CodeObjectView {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct CellViewObject {
    header: ObjectHeader,
    cell: Arc<Cell>,
}

impl CellViewObject {
    #[inline]
    pub fn new(cell: Arc<Cell>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::CELL_VIEW),
            cell,
        }
    }

    #[inline]
    pub fn cell(&self) -> &Arc<Cell> {
        &self.cell
    }
}

impl PyObject for CellViewObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct GenericAliasObject {
    header: ObjectHeader,
    origin: Value,
    args: Box<[Value]>,
}

impl GenericAliasObject {
    #[inline]
    pub fn new(origin: Value, args: Vec<Value>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::GENERIC_ALIAS),
            origin,
            args: args.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn origin(&self) -> Value {
        self.origin
    }

    #[inline]
    pub fn args(&self) -> &[Value] {
        &self.args
    }
}

impl PyObject for GenericAliasObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct UnionTypeObject {
    header: ObjectHeader,
    members: Box<[TypeId]>,
}

impl UnionTypeObject {
    #[inline]
    pub fn new(members: Vec<TypeId>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::UNION),
            members: members.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn members(&self) -> &[TypeId] {
        &self.members
    }
}

impl PyObject for UnionTypeObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct SingletonObject {
    header: ObjectHeader,
}

impl SingletonObject {
    #[inline]
    pub fn new(type_id: TypeId) -> Self {
        debug_assert!(matches!(
            type_id,
            TypeId::ELLIPSIS | TypeId::NOT_IMPLEMENTED
        ));
        Self {
            header: ObjectHeader::new(type_id),
        }
    }
}

impl PyObject for SingletonObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingProxySource {
    BuiltinType(TypeId),
    UserClass(usize),
    Dict(Value),
}

#[repr(C)]
pub struct MappingProxyObject {
    header: ObjectHeader,
    source: MappingProxySource,
}

impl MappingProxyObject {
    #[inline]
    pub fn for_builtin_type(type_id: TypeId) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MAPPING_PROXY),
            source: MappingProxySource::BuiltinType(type_id),
        }
    }

    #[inline]
    pub fn for_user_class(class: *const PyClassObject) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MAPPING_PROXY),
            source: MappingProxySource::UserClass(class as usize),
        }
    }

    #[inline]
    pub fn for_mapping(mapping: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MAPPING_PROXY),
            source: MappingProxySource::Dict(mapping),
        }
    }

    #[inline]
    pub fn source(&self) -> MappingProxySource {
        self.source
    }
}

impl PyObject for MappingProxyObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictViewKind {
    Keys,
    Values,
    Items,
}

impl DictViewKind {
    #[inline]
    pub const fn type_id(self) -> TypeId {
        match self {
            Self::Keys => TypeId::DICT_KEYS,
            Self::Values => TypeId::DICT_VALUES,
            Self::Items => TypeId::DICT_ITEMS,
        }
    }
}

#[repr(C)]
pub struct DictViewObject {
    header: ObjectHeader,
    dict: Value,
    kind: DictViewKind,
}

impl DictViewObject {
    #[inline]
    pub fn new(kind: DictViewKind, dict: Value) -> Self {
        Self {
            header: ObjectHeader::new(kind.type_id()),
            dict,
            kind,
        }
    }

    #[inline]
    pub fn dict(&self) -> Value {
        self.dict
    }

    #[inline]
    pub fn kind(&self) -> DictViewKind {
        self.kind
    }
}

impl PyObject for DictViewObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct DescriptorViewObject {
    header: ObjectHeader,
    owner: TypeId,
    name: InternedString,
}

impl DescriptorViewObject {
    #[inline]
    pub fn new(type_id: TypeId, owner: TypeId, name: InternedString) -> Self {
        debug_assert!(matches!(
            type_id,
            TypeId::WRAPPER_DESCRIPTOR
                | TypeId::METHOD_DESCRIPTOR
                | TypeId::CLASSMETHOD_DESCRIPTOR
                | TypeId::GETSET_DESCRIPTOR
                | TypeId::MEMBER_DESCRIPTOR
        ));
        Self {
            header: ObjectHeader::new(type_id),
            owner,
            name,
        }
    }

    #[inline]
    pub fn owner(&self) -> TypeId {
        self.owner
    }

    #[inline]
    pub fn name(&self) -> &InternedString {
        &self.name
    }
}

impl PyObject for DescriptorViewObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct MethodWrapperObject {
    header: ObjectHeader,
    owner: TypeId,
    name: InternedString,
    receiver: Value,
}

impl MethodWrapperObject {
    #[inline]
    pub fn new(owner: TypeId, name: InternedString, receiver: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::METHOD_WRAPPER),
            owner,
            name,
            receiver,
        }
    }

    #[inline]
    pub fn owner(&self) -> TypeId {
        self.owner
    }

    #[inline]
    pub fn name(&self) -> &InternedString {
        &self.name
    }

    #[inline]
    pub fn receiver(&self) -> Value {
        self.receiver
    }
}

impl PyObject for MethodWrapperObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct FrameViewObject {
    header: ObjectHeader,
    code: Option<Arc<CodeObject>>,
    globals: Value,
    locals: Value,
    line_number: u32,
    lasti: u32,
    back: Option<Value>,
}

impl FrameViewObject {
    #[inline]
    pub fn new(
        code: Option<Arc<CodeObject>>,
        globals: Value,
        locals: Value,
        line_number: u32,
        lasti: u32,
        back: Option<Value>,
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::FRAME),
            code,
            globals,
            locals,
            line_number,
            lasti,
            back,
        }
    }

    #[inline]
    pub fn code(&self) -> Option<&Arc<CodeObject>> {
        self.code.as_ref()
    }

    #[inline]
    pub fn globals(&self) -> Value {
        self.globals
    }

    #[inline]
    pub fn locals(&self) -> Value {
        self.locals
    }

    #[inline]
    pub fn line_number(&self) -> u32 {
        self.line_number
    }

    #[inline]
    pub fn lasti(&self) -> u32 {
        self.lasti
    }

    #[inline]
    pub fn back(&self) -> Option<Value> {
        self.back
    }
}

impl PyObject for FrameViewObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[repr(C)]
pub struct TracebackViewObject {
    header: ObjectHeader,
    frame: Value,
    next: Option<Value>,
    line_number: u32,
    lasti: u32,
}

impl TracebackViewObject {
    #[inline]
    pub fn new(frame: Value, next: Option<Value>, line_number: u32, lasti: u32) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TRACEBACK),
            frame,
            next,
            line_number,
            lasti,
        }
    }

    #[inline]
    pub fn frame(&self) -> Value {
        self.frame
    }

    #[inline]
    pub fn next(&self) -> Option<Value> {
        self.next
    }

    #[inline]
    pub fn set_next(&mut self, next: Option<Value>) {
        self.next = next;
    }

    #[inline]
    pub fn line_number(&self) -> u32 {
        self.line_number
    }

    #[inline]
    pub fn lasti(&self) -> u32 {
        self.lasti
    }
}

impl PyObject for TracebackViewObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
